"""El bucle de entrenamiento y evaluación: LOSO, modelo y líneas base en la misma partición.

Lo que este archivo garantiza, y que es lo que hace creíble un resultado:

1. **Se reparte por persona** y se valida dejando un sujeto fuera. Cada pliegue responde la
   pregunta que importa: cómo le va al modelo con alguien que nunca vio.
2. **El modelo y las cinco líneas base se miden en los mismos pliegues y sobre las mismas
   filas.** Si no, la comparación no es una comparación.
3. **Las métricas se agregan sobre todas las predicciones juntas**, no promediando los doce
   números por pliegue. Promediar porcentajes de muestras de distinto tamaño los pondera mal:
   un sujeto con 400 filas acabaría pesando lo mismo que uno con 4 000.
4. **Los hiperparámetros no se tocan mirando la prueba.** Este bucle no busca nada; la búsqueda
   va aparte y contra validación, porque ajustarlos mirando la prueba equivale a entrenar con
   la prueba y el número que sale de ahí no se puede reportar.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from src.eval.metrics import Resultado, evaluar
from src.eval.particion import pliegues_loso
from src.features.features import a_matriz
from src.features.ventanas import Fila


@dataclass(frozen=True)
class Prediccion:
    """Una predicción con lo necesario para medirla y para saber de quién es."""

    sujeto: str
    real: float
    predicho: float


@dataclass(frozen=True)
class Corrida:
    """Lo que deja una evaluación completa: el modelo y sus líneas base, ya medidos."""

    modelo: str
    horizonte_min: int
    n_entrenamiento: int
    resultado: Resultado
    lineas_base: dict[str, Resultado]
    por_sujeto: dict[str, Resultado]
    sujetos: tuple[str, ...]

    # Las predicciones de los pliegues, crudas. Se guardan porque una métrica agregada no deja
    # responder después nada que no estuviera previsto: la pregunta de a qué umbral conviene
    # avisar (`src/eval/umbral_de_aviso.py`) apareció al ver los resultados, y sin esto
    # costaba volver a entrenar los dieciocho pliegues —media hora— cada vez que se quisiera
    # mirar la curva desde otro sitio.
    predicciones: tuple[Prediccion, ...] = ()
    predicciones_base: dict[str, tuple[Prediccion, ...]] = field(default_factory=dict)

    @property
    def le_gana_a_la_persistencia(self) -> bool:
        """Lo primero que hay que mirar. **Si es falso, el modelo no justifica su existencia.**

        Se compara por RMSE, y además se exige que no empeore el A+B de la rejilla de Clarke:
        un RMSE mejor con más predicciones en zonas peligrosas no es mejor para una persona.
        """
        persistencia = self.lineas_base.get("persistencia")
        if persistencia is None:
            return False
        return (
            self.resultado.rmse < persistencia.rmse
            and self.resultado.clarke_ab >= persistencia.clarke_ab
        )


def _limpiar(predicciones: Sequence[Prediccion]) -> list[Prediccion]:
    """Tira las predicciones `NaN` —la extrapolación las produce al empezar la serie— y deja el
    resto. El `n` del `Resultado` delata cuántas quedaron, que es lo que permite leer el número.
    """
    return [p for p in predicciones if not (math.isnan(p.predicho) or math.isnan(p.real))]


def _medir(predicciones: Sequence[Prediccion], horizonte_min: int, umbral_hipo: int) -> Resultado:
    limpias = _limpiar(predicciones)
    if not limpias:
        raise ValueError("no quedó ninguna predicción que medir")
    return evaluar(
        [p.real for p in limpias],
        [p.predicho for p in limpias],
        horizonte_min=horizonte_min,
        umbral_hipo=umbral_hipo,
    )


def evaluar_con_loso(
    filas: Sequence[Fila],
    *,
    juego: str,
    horizonte_min: int,
    construir_modelo: Callable[[], object],
    construir_lineas_base: Callable[[], dict[str, object]] | None = None,
    umbral_hipo: int = 70,
    nombre: str = "modelo",
) -> Corrida:
    """Entrena y evalúa dejando un sujeto fuera, el modelo y sus líneas base a la vez.

    `construir_modelo` devuelve un modelo **nuevo** en cada llamada, y se llama una vez por
    pliegue: reutilizar el mismo objeto lo seguiría entrenando con los sujetos de los pliegues
    anteriores, que es fuga de información entre pliegues y da un resultado inflado sin que
    nada falle.
    """
    if not filas:
        raise ValueError("no hay filas con las que entrenar")

    predicciones: list[Prediccion] = []
    por_linea_base: dict[str, list[Prediccion]] = {}
    n_entrenamiento = 0

    for pliegue in pliegues_loso(filas):
        x_entrena = a_matriz([f.caracteristicas for f in pliegue.entrenamiento], juego)
        y_entrena = [f.objetivo for f in pliegue.entrenamiento]
        x_prueba = a_matriz([f.caracteristicas for f in pliegue.prueba], juego)
        reales = [f.objetivo for f in pliegue.prueba]
        sujeto = pliegue.sujetos_prueba[0]
        n_entrenamiento = max(n_entrenamiento, len(y_entrena))

        modelo = construir_modelo()
        modelo.ajustar(x_entrena, y_entrena)  # type: ignore[attr-defined]
        predicciones.extend(
            Prediccion(sujeto=sujeto, real=r, predicho=p)
            for r, p in zip(reales, modelo.predecir(x_prueba), strict=True)  # type: ignore[attr-defined]
        )

        for nombre_base, base in (construir_lineas_base or dict)().items():
            base.ajustar(x_entrena, y_entrena)  # type: ignore[attr-defined]
            por_linea_base.setdefault(nombre_base, []).extend(
                Prediccion(sujeto=sujeto, real=r, predicho=p)
                for r, p in zip(reales, base.predecir(x_prueba), strict=True)  # type: ignore[attr-defined]
            )

    sujetos = tuple(sorted({p.sujeto for p in predicciones}))
    return Corrida(
        modelo=nombre,
        horizonte_min=horizonte_min,
        n_entrenamiento=n_entrenamiento,
        resultado=_medir(predicciones, horizonte_min, umbral_hipo),
        lineas_base={n: _medir(ps, horizonte_min, umbral_hipo) for n, ps in por_linea_base.items()},
        por_sujeto=_por_sujeto(predicciones, horizonte_min, umbral_hipo),
        sujetos=sujetos,
        predicciones=tuple(predicciones),
        predicciones_base={n: tuple(ps) for n, ps in por_linea_base.items()},
    )


def _por_sujeto(
    predicciones: Sequence[Prediccion], horizonte_min: int, umbral_hipo: int
) -> dict[str, Resultado]:
    """Las métricas de cada persona por separado.

    Van al reporte porque un promedio esconde a la persona para la que el sistema funciona
    peor, y con doce sujetos eso es justo lo que hay que poder mirar (riesgo R-2). Un sujeto
    cuyas filas se quedaron todas en `NaN` se omite en lugar de reventar la corrida entera.
    """
    salida: dict[str, Resultado] = {}
    for sujeto in sorted({p.sujeto for p in predicciones}):
        suyas = [p for p in predicciones if p.sujeto == sujeto]
        try:
            salida[sujeto] = _medir(suyas, horizonte_min, umbral_hipo)
        except ValueError:
            continue
    return salida


def resumen(corrida: Corrida) -> str:
    """El cuadro que se imprime al terminar y se pega en el documento.

    Ordenado con el modelo arriba y las líneas base debajo, y con una última línea que dice si
    le gana a la persistencia, porque es la pregunta que decide si el modelo sirve.
    """
    encabezado = (
        f"{'':22} {'n':>7} {'RMSE':>8} {'MAE':>8} {'MARD':>7} {'A+B':>7} "
        f"{'Sens.hipo':>10} {'Falsas':>8}"
    )
    lineas = [
        f"{corrida.modelo} a {corrida.horizonte_min} min · "
        f"{len(corrida.sujetos)} sujetos · {corrida.n_entrenamiento} filas de entrenamiento",
        "",
        encabezado,
        "-" * len(encabezado),
        _fila_del_resumen(corrida.modelo, corrida.resultado),
    ]
    for nombre, resultado in corrida.lineas_base.items():
        lineas.append(_fila_del_resumen(nombre, resultado))
    lineas += [
        "",
        (f"Le gana a la persistencia: {'sí' if corrida.le_gana_a_la_persistencia else 'NO'}"),
    ]
    return "\n".join(lineas)


def _fila_del_resumen(nombre: str, r: Resultado) -> str:
    sensibilidad = "no se midió" if r.sensibilidad_hipo is None else f"{r.sensibilidad_hipo:.1f}"
    return (
        f"{nombre:22} {r.n:>7} {r.rmse:>8.2f} {r.mae:>8.2f} {r.mard:>6.2f}% "
        f"{r.clarke_ab:>6.2f}% {sensibilidad:>10} {r.falsas_alarmas:>7.1f}%"
    )


@dataclass(frozen=True)
class ResultadoSimple:
    """RMSE y MAE y nada más, para un objetivo que **no es una glucosa**.

    Existe porque M-3 predice también *cuándo* ocurre el pico, en minutos, y las métricas
    clínicas no se le pueden aplicar: la rejilla de Clarke clasifica el error de una glucosa
    por su consecuencia, y «zona A» sobre una cantidad de minutos no significa nada. Pasar
    minutos por `evaluar` devolvería números con nombre clínico y sin sentido, que es peor que
    no tenerlos.
    """

    n: int
    unidad: str
    rmse: float
    mae: float


def evaluar_con_loso_simple(
    filas: Sequence[object],
    *,
    juego: str,
    objetivo: Callable[[object], float],
    construir_modelo: Callable[[], object],
    unidad: str,
) -> tuple[ResultadoSimple, dict[str, ResultadoSimple]]:
    """El mismo LOSO, midiendo solo RMSE y MAE. Devuelve el agregado y el de cada sujeto.

    `objetivo` saca el número a predecir de cada fila, para que la misma función sirva para el
    valor del pico y para su hora sin duplicar el bucle.
    """
    from src.eval.metrics import mae, rmse

    if not filas:
        raise ValueError("no hay filas con las que entrenar")

    predicciones: list[Prediccion] = []
    for pliegue in pliegues_loso(filas):  # type: ignore[arg-type]
        x_entrena = a_matriz([f.caracteristicas for f in pliegue.entrenamiento], juego)  # type: ignore[attr-defined]
        x_prueba = a_matriz([f.caracteristicas for f in pliegue.prueba], juego)  # type: ignore[attr-defined]
        modelo = construir_modelo()
        modelo.ajustar(x_entrena, [objetivo(f) for f in pliegue.entrenamiento])  # type: ignore[attr-defined]
        predicciones.extend(
            Prediccion(sujeto=pliegue.sujetos_prueba[0], real=objetivo(f), predicho=p)
            for f, p in zip(pliegue.prueba, modelo.predecir(x_prueba), strict=True)  # type: ignore[attr-defined]
        )

    def medir(ps: Sequence[Prediccion]) -> ResultadoSimple:
        limpias = _limpiar(ps)
        return ResultadoSimple(
            n=len(limpias),
            unidad=unidad,
            rmse=rmse([p.real for p in limpias], [p.predicho for p in limpias]),
            mae=mae([p.real for p in limpias], [p.predicho for p in limpias]),
        )

    por_sujeto = {
        s: medir([p for p in predicciones if p.sujeto == s])
        for s in sorted({p.sujeto for p in predicciones})
    }
    return medir(predicciones), por_sujeto
