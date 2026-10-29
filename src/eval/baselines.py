"""Las líneas base obligatorias. **Un modelo que no le gana a la persistencia no justifica su
existencia**, y ese es el punto de este archivo.

Las cinco del protocolo son persistencia, extrapolación lineal, Ridge, Random Forest y LSTM.
Aquí van las **dos que no necesitan entrenarse**: se calculan sobre la serie misma, así que
están escritas y probadas antes de que llegue el conjunto de datos. Las otras tres llegan con
`src/train/`, cuando haya con qué ajustarlas.

Las dos de aquí son las que de verdad humillan a un modelo mal hecho. La glucosa tiene mucha
inercia: a 30 minutos, «va a seguir igual» acierta notablemente, y es el número contra el que
hay que comparar antes de celebrar un RMSE de 18 mg/dL.
"""

import math
from collections.abc import Sequence

NADA = math.nan


def persistencia(serie: Sequence[float], pasos_horizonte: int) -> list[float]:
    """«La glucosa dentro de N pasos será la de ahora».

    Devuelve una predicción **por cada punto que tiene un valor real con el que compararse**,
    es decir `len(serie) - pasos_horizonte`. El real correspondiente es
    `serie[pasos_horizonte:]`, y `pares_con_reales` arma los dos de una vez para que nadie los
    desalinee a mano: desplazar por uno estas dos listas cambia todas las métricas sin que
    nada falle.
    """
    if pasos_horizonte <= 0:
        raise ValueError("el horizonte debe ser de al menos un paso")
    if len(serie) <= pasos_horizonte:
        return []
    return list(serie[:-pasos_horizonte])


def extrapolacion_lineal(
    serie: Sequence[float], pasos_horizonte: int, pasos_pendiente: int = 3
) -> list[float]:
    """«Va a seguir subiendo al ritmo que lleva».

    Toma la pendiente de los últimos `pasos_pendiente` pasos y la proyecta. Es la línea base
    que de verdad compite en las subidas posprandiales, donde la persistencia se queda corta.

    Los primeros puntos de la serie salen `NaN` porque no tienen pendiente detrás: **no se
    rellenan con el valor actual**, porque eso convertiría en silencio esta línea base en la
    de persistencia justo en los puntos donde la serie empieza, y haría ver a la
    extrapolación mejor de lo que es.
    """
    if pasos_horizonte <= 0:
        raise ValueError("el horizonte debe ser de al menos un paso")
    if pasos_pendiente <= 0:
        raise ValueError("la pendiente necesita al menos un paso")
    if len(serie) <= pasos_horizonte:
        return []

    predichos: list[float] = []
    for i in range(len(serie) - pasos_horizonte):
        if i < pasos_pendiente:
            predichos.append(NADA)
            continue
        pendiente = (serie[i] - serie[i - pasos_pendiente]) / pasos_pendiente
        predichos.append(serie[i] + pendiente * pasos_horizonte)
    return predichos


def pares_con_reales(
    serie: Sequence[float], predichos: Sequence[float], pasos_horizonte: int
) -> tuple[list[float], list[float]]:
    """Alinea una línea base con sus valores reales y tira los pares con `NaN`.

    Es la única forma correcta de comparar: el real del punto `i` es `serie[i + horizonte]`.
    Hacerlo a mano en cada evaluación es el error de un solo índice que desplaza todas las
    métricas, y aquí está escrito una vez.

    Los pares con `NaN` se descartan **y el largo devuelto lo delata**: una línea base que
    solo pudo predecir la mitad de los puntos no se puede comparar de igual a igual contra
    una que predijo todos, y el reporte tiene que decir sobre cuántos pares se midió cada
    una (de ahí que `Resultado` lleve su `n`).
    """
    reales = list(serie[pasos_horizonte:])
    if len(reales) != len(predichos):
        raise ValueError(
            f"la línea base dio {len(predichos)} predicciones y hay {len(reales)} reales: "
            "alinear mal desplaza todas las métricas"
        )
    limpios = [
        (r, p)
        for r, p in zip(reales, predichos, strict=True)
        if not (math.isnan(p) or math.isnan(r))
    ]
    if not limpios:
        return [], []
    return [r for r, _ in limpios], [p for _, p in limpios]
