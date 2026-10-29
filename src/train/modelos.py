"""Los modelos y las líneas base, todos detrás de la misma interfaz.

Una sola interfaz para que **la comparación sea sobre las mismas filas**. Es el punto entero:
«RMSE 18 mg/dL» no significa nada suelto, y la forma de acabar comparando contra un número de
otra corrida es tener el modelo en un archivo y las líneas base en otro. Aquí se ajustan y se
evalúan en el mismo bucle, sobre la misma partición.

Las cinco líneas base del protocolo:

| Línea base | Qué dice | Hay que ajustarla |
| --- | --- | --- |
| Persistencia | «dentro de 30 min será igual que ahora» | no |
| Extrapolación lineal | «va a seguir al ritmo que lleva» | no |
| Ridge | regresión lineal con regularización | sí |
| Random Forest | bosque de árboles | sí |
| LSTM | red recurrente | sí, y necesita un marco de redes |

**La persistencia es la que importa.** La glucosa tiene mucha inercia: a 30 minutos, «va a
seguir igual» acierta notablemente, y es el número contra el que hay que comparar antes de
celebrar cualquier cosa. **Un modelo que no le gana a la persistencia no justifica su
existencia.**

## Los valores que faltan, y por qué cada modelo los trata distinto

Las filas traen `NaN` donde no hay dato —un rezago de 60 minutos al empezar la serie, por
ejemplo— y eso es información, no un cero.

- **XGBoost y Random Forest por histograma** aprenden para qué lado mandar un `NaN`. Lo reciben
  tal cual.
- **Ridge no puede con `NaN`**, así que su tubería le pone la mediana de la columna. Es una
  desventaja real de la línea base y **está documentada a propósito**: si Ridge sale mal, parte
  de la culpa puede ser la imputación, y eso hay que poder decirlo en lugar de concluir que el
  modelo lineal no sirve.
"""

import math
from collections.abc import Sequence
from typing import Any, Protocol

from src.features.features import CARACTERISTICAS_DENSAS

Matriz = Sequence[Sequence[float]]
Vector = Sequence[float]


class Modelo(Protocol):
    """Lo mínimo para entrar al bucle de evaluación."""

    nombre: str
    algoritmo: str

    @property
    def hiperparametros(self) -> dict[str, Any]: ...

    def ajustar(self, x: Matriz, y: Vector) -> None: ...

    def predecir(self, x: Matriz) -> list[float]: ...


# --- Las dos que no se ajustan --------------------------------------------------------------


class Persistencia:
    """«Dentro de N minutos será igual que ahora»: devuelve `mg_dl` de la propia fila.

    No mira ninguna otra columna y no se entrena, así que su `ajustar` no hace nada. Se escribe
    como modelo y no como función aparte para que se evalúe en el mismo bucle y sobre las mismas
    filas que todo lo demás; comparar contra una persistencia calculada en otro sitio, sobre
    otro conjunto de pares, es la trampa que este diseño evita.
    """

    nombre = "persistencia"
    algoritmo = "persistencia"

    def __init__(self, caracteristicas: Sequence[str] = CARACTERISTICAS_DENSAS) -> None:
        self._i = list(caracteristicas).index("mg_dl")

    @property
    def hiperparametros(self) -> dict[str, Any]:
        return {}

    def ajustar(self, x: Matriz, y: Vector) -> None:
        return None

    def predecir(self, x: Matriz) -> list[float]:
        return [float(fila[self._i]) for fila in x]


class ExtrapolacionLineal:
    """«Va a seguir al ritmo que lleva»: `mg_dl + pendiente * horizonte`.

    Es la que de verdad compite en las subidas posprandiales, donde la persistencia se queda
    corta.

    ## Sobre qué ventana se mide la pendiente, que importa más de lo que parece

    La pendiente sale de `(mg_dl - rezago_N) / N`, y **N cambia la línea base por un factor de
    tres**. Medido sobre las series de prueba, extrapolando a 30 minutos:

    | Ventana de la pendiente | RMSE de la línea base |
    | --- | --- |
    | 5 min (`derivada_1`) | 53.1 |
    | 15 min | 26.5 |
    | **30 min (la que se usa)** | **20.0** |
    | 60 min | 17.9 |

    La pendiente de cinco minutos es casi toda ruido del sensor, y multiplicarla por treinta lo
    amplifica por seis: da una línea base pésima. **Y una línea base pésima adula al modelo**,
    que es justo la dirección en la que no conviene equivocarse, porque el proyecto se sostiene
    en «le gana a la persistencia y a la extrapolación».

    Por omisión la ventana es igual al horizonte, que es lo habitual en la literatura y lo
    defendible: «el ritmo que lleva» medido en el mismo plazo que se proyecta. Se puede cambiar,
    y el valor queda en los hiperparámetros del manifiesto para que el reporte diga cuál se usó.

    Cuando el rezago que hace falta es `NaN` —la serie acaba de empezar— devuelve `NaN` en lugar
    de caer a la persistencia: rellenar la convertiría en silencio en la otra línea base y la
    haría ver mejor de lo que es.
    """

    nombre = "extrapolacion_lineal"
    algoritmo = "extrapolacion_lineal"

    def __init__(
        self,
        horizonte_min: int,
        pendiente_min: int | None = None,
        caracteristicas: Sequence[str] = CARACTERISTICAS_DENSAS,
    ) -> None:
        nombres = list(caracteristicas)
        self._horizonte = horizonte_min
        self._pendiente_min = pendiente_min if pendiente_min is not None else horizonte_min
        rezago = f"rezago_{self._pendiente_min}"
        if rezago not in nombres:
            disponibles = sorted(
                int(n.removeprefix("rezago_")) for n in nombres if n.startswith("rezago_")
            )
            raise ValueError(
                f"no hay {rezago} entre las características; las ventanas disponibles son "
                f"{disponibles} minutos"
            )
        self._i_valor = nombres.index("mg_dl")
        self._i_rezago = nombres.index(rezago)

    @property
    def hiperparametros(self) -> dict[str, Any]:
        return {"horizonte_min": self._horizonte, "pendiente_min": self._pendiente_min}

    def ajustar(self, x: Matriz, y: Vector) -> None:
        return None

    def predecir(self, x: Matriz) -> list[float]:
        salida: list[float] = []
        for fila in x:
            valor, atras = float(fila[self._i_valor]), float(fila[self._i_rezago])
            if math.isnan(atras):
                salida.append(math.nan)
                continue
            pendiente = (valor - atras) / self._pendiente_min
            salida.append(valor + pendiente * self._horizonte)
        return salida


# --- Las que se ajustan ---------------------------------------------------------------------


class XGBoost:
    """El modelo del proyecto: regresión con árboles potenciados por gradiente.

    Regresión y no clasificación: se predice **el valor** y la alerta se deriva comparándolo
    con el umbral del perfil (RF-26). Así la persona ve la curva y entiende por qué se disparó
    la alerta, en lugar de recibir un sí o un no sin contexto.

    Los hiperparámetros de arranque son conservadores: con doce sujetos, un modelo profundo
    memoriza a las personas del entrenamiento y la búsqueda se hace **solo contra validación**,
    nunca mirando la prueba.
    """

    nombre = "xgboost"
    algoritmo = "xgboost"

    POR_OMISION: dict[str, Any] = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "reg_lambda": 1.0,
        "objective": "reg:squarederror",
    }

    def __init__(self, semilla: int, **sobre: Any) -> None:
        self._params = {**self.POR_OMISION, **sobre, "random_state": semilla}
        self._modelo: Any = None

    @property
    def hiperparametros(self) -> dict[str, Any]:
        return dict(self._params)

    def ajustar(self, x: Matriz, y: Vector) -> None:
        from xgboost import XGBRegressor

        self._modelo = XGBRegressor(**self._params)
        self._modelo.fit(_np(x), _np(y))

    def predecir(self, x: Matriz) -> list[float]:
        if self._modelo is None:
            raise RuntimeError("hay que ajustar el modelo antes de predecir")
        return [float(v) for v in self._modelo.predict(_np(x))]

    def guardar(self, base) -> str:
        """Guarda en el formato binario propio de XGBoost, que es el que lee el backend."""
        if self._modelo is None:
            raise RuntimeError("no hay modelo que guardar")
        ruta = f"{base}.ubj"
        self._modelo.save_model(ruta)
        return ruta


class Ridge:
    """Línea base lineal con regularización. Recibe la mediana donde la fila trae `NaN`.

    La imputación es parte de la línea base y no del modelo del proyecto, y conviene recordarlo
    al leer su número: si Ridge sale mal, parte puede ser la mediana y no la linealidad.
    """

    nombre = "ridge"
    algoritmo = "ridge"

    def __init__(self, semilla: int, alpha: float = 1.0) -> None:
        self._alpha = alpha
        self._semilla = semilla
        self._tuberia: Any = None

    @property
    def hiperparametros(self) -> dict[str, Any]:
        return {"alpha": self._alpha, "imputacion": "mediana", "escalado": "estandar"}

    def ajustar(self, x: Matriz, y: Vector) -> None:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge as RidgeSk
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self._tuberia = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            RidgeSk(alpha=self._alpha, random_state=self._semilla),
        )
        self._tuberia.fit(_np(x), _np(y))

    def predecir(self, x: Matriz) -> list[float]:
        if self._tuberia is None:
            raise RuntimeError("hay que ajustar el modelo antes de predecir")
        return [float(v) for v in self._tuberia.predict(_np(x))]


class Bosque:
    """Línea base de bosque aleatorio, por histograma para que acepte `NaN` como los árboles
    potenciados y la comparación no quede sesgada por la imputación.
    """

    nombre = "bosque"
    algoritmo = "random_forest"

    def __init__(self, semilla: int, **sobre: Any) -> None:
        self._params = {"max_iter": 300, "max_depth": 6, **sobre, "random_state": semilla}
        self._modelo: Any = None

    @property
    def hiperparametros(self) -> dict[str, Any]:
        return dict(self._params)

    def ajustar(self, x: Matriz, y: Vector) -> None:
        from sklearn.ensemble import HistGradientBoostingRegressor

        self._modelo = HistGradientBoostingRegressor(**self._params)
        self._modelo.fit(_np(x), _np(y))

    def predecir(self, x: Matriz) -> list[float]:
        if self._modelo is None:
            raise RuntimeError("hay que ajustar el modelo antes de predecir")
        return [float(v) for v in self._modelo.predict(_np(x))]


def _np(datos):
    """A `numpy`, importado aquí dentro.

    Igual que con xgboost y scikit-learn: así importar este módulo no obliga a tener instaladas
    las bibliotecas del entrenamiento, y las pruebas de las dos líneas base que no se ajustan
    corren en la integración continua, que solo instala pytest y ruff.
    """
    import numpy as np

    return np.asarray(datos, dtype=float)


# La línea base que falta, y por qué no está:
#
# **LSTM.** El protocolo la pide y no está escrita. Necesita un marco de redes —PyTorch o
# Keras—, que son cientos de megabytes y una decisión que conviene tomar con los datos delante:
# con doce sujetos y ocho semanas cada uno puede que una red recurrente no se sostenga, y eso
# se sabe midiendo, no antes. Se agrega en el paso que la implemente, junto con su dependencia.
# Mientras no esté, **el reporte tiene que decir que falta**, no omitirla de la tabla.
FALTA_LSTM = "necesita un marco de redes; se decide con los datos delante"
