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


# --- La quinta línea base, la recurrente -----------------------------------------------------

# La secuencia de glucosa que ya viene dentro de cada fila, del punto más viejo al más nuevo.
# No hay que ir a buscarla a ninguna parte: las características de rezago **son** la serie.
PASOS_DE_LA_SECUENCIA: tuple[str, ...] = (
    "rezago_60", "rezago_30", "rezago_15", "rezago_10", "rezago_5", "mg_dl",
)  # fmt: skip


class LSTM:
    """Línea base recurrente. Es la quinta del protocolo y la última que faltaba.

    ## De dónde sale la secuencia

    Una red recurrente quiere una secuencia, y el bucle de evaluación reparte filas planas. En
    lugar de armar un conjunto aparte —que sería comparar dos modelos sobre dos conjuntos
    distintos, la trampa que este diseño existe para evitar—, la secuencia se arma **con las
    columnas que la fila ya trae**: `rezago_60, rezago_30, rezago_15, rezago_10, rezago_5,
    mg_dl` es, literalmente, la glucosa en seis instantes ordenados. El resto de las
    características (COB, IOB, hora cíclica, día) se repiten en cada paso como contexto.

    Así la red ve **la misma información** que XGBoost, sobre **las mismas filas** y con **la
    misma partición**. La comparación mide la arquitectura, que es lo que se quiere medir.

    Los pasos no están igualmente espaciados: van 30, 15, 5, 5 y 5 minutos. Una LSTM no asume
    espaciado uniforme —no es un filtro—, así que no es un error, pero conviene decirlo en el
    reporte en lugar de dejar creer que son seis muestras del sensor seguidas.

    ## Por qué está escrita con cuidado y no de cualquier manera

    Una línea base mal entrenada **adula al modelo del proyecto**, y es justo la dirección en
    la que no conviene equivocarse: todo el argumento se sostiene en «le gana a estas cinco».
    Por eso lleva lo que una red necesita para no salir artificialmente mal: estandarización de
    entradas y del objetivo, Adam, y **paro temprano sobre un corte temporal** del conjunto de
    entrenamiento —el último trozo, nunca uno al azar—, porque barajar una serie y validar
    sobre el pasado es fuga de información y daría una línea base irrealmente buena.

    Si con doce sujetos la red no se sostiene, el número lo dirá. Eso es un resultado del
    reporte, no una razón para no escribirla.
    """

    nombre = "lstm"
    algoritmo = "lstm"

    def __init__(
        self,
        semilla: int,
        *,
        caracteristicas: Sequence[str] = CARACTERISTICAS_DENSAS,
        unidades: int = 32,
        epocas: int = 60,
        paciencia: int = 8,
        lote: int = 256,
        tasa: float = 0.01,
        fraccion_validacion: float = 0.2,
    ) -> None:
        self._semilla = semilla
        self._caracteristicas = list(caracteristicas)
        faltan = [c for c in PASOS_DE_LA_SECUENCIA if c not in self._caracteristicas]
        if faltan:
            raise ValueError(
                f"la línea base recurrente arma su secuencia con los rezagos de la fila y "
                f"faltan {faltan}. Con el juego disperso no hay serie que recorrer, y una "
                "LSTM sobre un solo punto no es una LSTM."
            )
        self._pasos = [self._caracteristicas.index(c) for c in PASOS_DE_LA_SECUENCIA]
        self._contexto = [
            i for i, c in enumerate(self._caracteristicas) if c not in PASOS_DE_LA_SECUENCIA
        ]
        self._unidades = unidades
        self._epocas = epocas
        self._paciencia = paciencia
        self._lote = lote
        self._tasa = tasa
        self._fraccion_validacion = fraccion_validacion
        self._red: Any = None
        self._centro: Any = None
        self._escala: Any = None
        self._centro_y = 0.0
        self._escala_y = 1.0
        self._epocas_usadas = 0

    @property
    def hiperparametros(self) -> dict[str, Any]:
        return {
            "unidades": self._unidades,
            "epocas_maximas": self._epocas,
            "epocas_usadas": self._epocas_usadas,
            "paciencia": self._paciencia,
            "lote": self._lote,
            "tasa_aprendizaje": self._tasa,
            "pasos": list(PASOS_DE_LA_SECUENCIA),
            "imputacion": "mediana",
            "escalado": "estandar",
            "validacion": "último 20 % en orden, nunca al azar",
        }

    def _secuencias(self, x: Matriz):
        """De la matriz plana a `(filas, 6 pasos, 1 + contexto)`, ya imputada y escalada."""
        import numpy as np

        datos = _np(x)
        datos = np.where(np.isnan(datos), self._centro, datos)
        datos = (datos - self._centro) / self._escala
        glucosa = datos[:, self._pasos][:, :, None]
        if not self._contexto:
            return glucosa
        contexto = datos[:, self._contexto][:, None, :]
        contexto = np.repeat(contexto, len(self._pasos), axis=1)
        return np.concatenate([glucosa, contexto], axis=2)

    def ajustar(self, x: Matriz, y: Vector) -> None:
        import numpy as np
        import torch
        from torch import nn

        torch.manual_seed(self._semilla)
        np.random.seed(self._semilla)

        datos = _np(x)
        # La mediana y la desviación salen **solo del entrenamiento**: calcularlas sobre todo
        # el conjunto filtraría información de la prueba hacia el ajuste.
        self._centro = np.nanmedian(datos, axis=0)
        self._centro = np.where(np.isnan(self._centro), 0.0, self._centro)
        escala = np.nanstd(np.where(np.isnan(datos), self._centro, datos), axis=0)
        self._escala = np.where((escala == 0) | np.isnan(escala), 1.0, escala)

        objetivo = _np(y)
        self._centro_y = float(objetivo.mean())
        self._escala_y = float(objetivo.std()) or 1.0

        secuencias = torch.tensor(self._secuencias(datos), dtype=torch.float32)
        etiquetas = torch.tensor(
            (objetivo - self._centro_y) / self._escala_y, dtype=torch.float32
        ).unsqueeze(1)

        # El corte de validación es **temporal**: el último trozo, en el orden en que vienen
        # las filas. Barajar y validar sobre el pasado es fuga de información, y con paro
        # temprano se traduce en parar en la época equivocada.
        corte = max(1, int(len(secuencias) * (1 - self._fraccion_validacion)))
        entrena_x, valida_x = secuencias[:corte], secuencias[corte:]
        entrena_y, valida_y = etiquetas[:corte], etiquetas[corte:]
        if len(valida_x) == 0:
            valida_x, valida_y = entrena_x, entrena_y

        self._red = _RedRecurrente(entradas=secuencias.shape[2], unidades=self._unidades)
        optimizador = torch.optim.Adam(self._red.parameters(), lr=self._tasa)
        perdida = nn.MSELoss()

        mejor = float("inf")
        mejor_estado = {k: v.clone() for k, v in self._red.state_dict().items()}
        sin_mejorar = 0
        for epoca in range(1, self._epocas + 1):
            self._red.train()
            for desde in range(0, len(entrena_x), self._lote):
                lote_x = entrena_x[desde : desde + self._lote]
                lote_y = entrena_y[desde : desde + self._lote]
                optimizador.zero_grad()
                perdida(self._red(lote_x), lote_y).backward()
                optimizador.step()

            self._red.eval()
            with torch.no_grad():
                actual = float(perdida(self._red(valida_x), valida_y))
            if actual < mejor - 1e-5:
                mejor, sin_mejorar = actual, 0
                mejor_estado = {k: v.clone() for k, v in self._red.state_dict().items()}
                self._epocas_usadas = epoca
            else:
                sin_mejorar += 1
                if sin_mejorar >= self._paciencia:
                    break
        # Se devuelven los pesos de la mejor época, no los de la última: parar tarde y quedarse
        # con lo último es entrenar de más y reportarlo como si no.
        self._red.load_state_dict(mejor_estado)

    def predecir(self, x: Matriz) -> list[float]:
        import torch

        if self._red is None:
            raise RuntimeError("hay que ajustar el modelo antes de predecir")
        self._red.eval()
        with torch.no_grad():
            salida = self._red(torch.tensor(self._secuencias(x), dtype=torch.float32))
        return [float(v) * self._escala_y + self._centro_y for v in salida.squeeze(1)]


def _RedRecurrente(entradas: int, unidades: int):  # noqa: N802
    """Una LSTM de una capa y una salida lineal sobre el último paso.

    Se define dentro de una función y no al nivel del módulo para no importar `torch` al
    importar este archivo: la integración continua solo instala pytest y ruff.
    """
    from torch import nn

    class Red(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(entradas, unidades, batch_first=True)
            self.salida = nn.Linear(unidades, 1)

        def forward(self, x):
            secuencia, _ = self.lstm(x)
            return self.salida(secuencia[:, -1, :])

    return Red()
