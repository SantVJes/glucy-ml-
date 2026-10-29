"""Las métricas del proyecto y sus metas, en un solo lugar.

Los números tienen que coincidir con `referencia/reglas_clinicas.py` de `glucy-backend`, que es
el árbitro: ahí están implementadas con 30 pruebas pasando, y `tests/test_metricas.py` comprueba
caso por caso que estas den lo mismo.

**Las metas buenas son 85 % de sensibilidad y 15 % de falsas alarmas.** El documento de la fase
F3 en Drive trae 80 % y 20 %; es una errata conocida y los valores del reporte son estos.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from src.eval.clarke import porcentaje_ab, reparto

# Las metas, para que una corrida pueda decir sola si pasó o no.
META_CLARKE_AB = 95.0
META_SENSIBILIDAD_HIPO = 85.0
META_FALSAS_ALARMAS_MAX = 15.0
META_RMSE_30_MAX = 20.0
META_MAE_30_MAX = 15.0
META_MARD_MAX = 12.0
META_ERROR_PICO_MAX = 25.0

UMBRAL_HIPO_DEFECTO = 70


def rmse(reales: Sequence[float], predichos: Sequence[float]) -> float:
    n = len(reales)
    if n == 0:
        raise ValueError("no hay pares que medir")
    return math.sqrt(sum((r - p) ** 2 for r, p in zip(reales, predichos, strict=True)) / n)


def mae(reales: Sequence[float], predichos: Sequence[float]) -> float:
    n = len(reales)
    if n == 0:
        raise ValueError("no hay pares que medir")
    return sum(abs(r - p) for r, p in zip(reales, predichos, strict=True)) / n


def mard(reales: Sequence[float], predichos: Sequence[float]) -> float:
    """Error absoluto relativo medio, en por ciento: es con lo que se comparan los sensores.

    Divide entre el valor real, así que un real de cero lo haría explotar. No se comprueba
    aquí porque una glucosa de cero no pasa la validación de 20–600 mg/dL mucho antes de
    llegar a una métrica; si llegara, el `ZeroDivisionError` es mejor que un promedio con un
    infinito escondido dentro.
    """
    n = len(reales)
    if n == 0:
        raise ValueError("no hay pares que medir")
    return 100 * sum(abs(r - p) / r for r, p in zip(reales, predichos, strict=True)) / n


def sensibilidad_hipoglucemia(
    reales: Sequence[float], predichos: Sequence[float], umbral: int = UMBRAL_HIPO_DEFECTO
) -> float:
    """De los episodios reales por debajo del umbral, cuántos se anticiparon. Meta ≥ 85 %.

    Levanta `ValueError` si no hubo ningún episodio real, **en lugar de devolver 100 %**: un
    conjunto de prueba sin hipoglucemias no demuestra que el modelo las detecte, y un 100 %
    en esa casilla del reporte sería la cifra más engañosa de todo el documento.
    """
    episodios = [(r, p) for r, p in zip(reales, predichos, strict=True) if r < umbral]
    if not episodios:
        raise ValueError("no hay episodios reales de hipoglucemia en el conjunto")
    return 100 * sum(1 for _, p in episodios if p < umbral) / len(episodios)


def tasa_falsas_alarmas(
    reales: Sequence[float], predichos: Sequence[float], umbral: int = UMBRAL_HIPO_DEFECTO
) -> float:
    """De todas las alertas que se habrían emitido, cuántas no correspondían. Meta ≤ 15 %.

    Es criterio de aceptación y no un adorno: si la app avisa de más, la persona deja de
    hacerle caso y el sistema pierde todo su valor, aunque las demás métricas salgan bien.

    Sin ninguna alerta devuelve 0.0, que aquí sí es honesto: no se emitió ninguna falsa
    porque no se emitió ninguna, y la sensibilidad de arriba es la que delata ese caso.
    """
    alertas = [(r, p) for r, p in zip(reales, predichos, strict=True) if p < umbral]
    if not alertas:
        return 0.0
    return 100 * sum(1 for r, _ in alertas if r >= umbral) / len(alertas)


@dataclass(frozen=True)
class Resultado:
    """Lo que deja una evaluación, y lo que se guarda junto al artefacto (RF-25).

    Lleva `n` porque una métrica sin el tamaño de la muestra no se puede leer: un 96 % de A+B
    sobre 200 pares y sobre 200 000 no dicen lo mismo. Y lleva `horizonte_min` porque las
    metas de RMSE y MAE son las de 30 minutos; a 60 el error es naturalmente mayor y
    compararlo contra la meta de 30 sería castigar al modelo por algo que no es su culpa.
    """

    n: int
    horizonte_min: int
    rmse: float
    mae: float
    mard: float
    clarke_ab: float
    clarke_reparto: dict[str, float]
    sensibilidad_hipo: float | None
    falsas_alarmas: float

    @property
    def cumple_criterio_principal(self) -> bool:
        """A+B, sensibilidad y falsas alarmas: los tres que deciden si el modelo sirve.

        `None` en la sensibilidad **no cuenta como cumplir**: significa que el conjunto no
        traía hipoglucemias y que ese criterio no se midió.
        """
        return (
            self.clarke_ab >= META_CLARKE_AB
            and self.sensibilidad_hipo is not None
            and self.sensibilidad_hipo >= META_SENSIBILIDAD_HIPO
            and self.falsas_alarmas <= META_FALSAS_ALARMAS_MAX
        )


def evaluar(
    reales: Sequence[float],
    predichos: Sequence[float],
    *,
    horizonte_min: int,
    umbral_hipo: int = UMBRAL_HIPO_DEFECTO,
) -> Resultado:
    """Todas las métricas de una corrida de una vez.

    La sensibilidad queda en `None` cuando el conjunto no trae hipoglucemias reales, y el
    reporte tiene que decir «no se midió» en lugar de dejar la casilla vacía o poner un cero
    que se lee como «no detectó ninguna».
    """
    if not reales:
        raise ValueError("no hay pares que medir")
    try:
        sensibilidad = sensibilidad_hipoglucemia(reales, predichos, umbral_hipo)
    except ValueError:
        sensibilidad = None

    return Resultado(
        n=len(reales),
        horizonte_min=horizonte_min,
        rmse=rmse(reales, predichos),
        mae=mae(reales, predichos),
        mard=mard(reales, predichos),
        clarke_ab=porcentaje_ab(reales, predichos),
        clarke_reparto=reparto(reales, predichos),
        sensibilidad_hipo=sensibilidad,
        falsas_alarmas=tasa_falsas_alarmas(reales, predichos, umbral_hipo),
    )
