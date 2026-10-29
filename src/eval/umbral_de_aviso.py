"""El umbral con el que se dispara el aviso, que no es el mismo que el umbral clínico.

Glucy predice un valor y **deriva la alerta comparándolo con un umbral** (RF-26 y siguientes).
Hasta aquí ese umbral era el clínico, 70 mg/dL, el mismo con el que se define qué es una
hipoglucemia de verdad. Usar el mismo número para las dos cosas tiene una consecuencia que no
se ve hasta que se miden dos modelos juntos: **compara dos modelos en puntos de operación
distintos y la comparación no vale**.

Un modelo de regresión entrenado con error cuadrático encoge hacia la media, porque encoger es
lo óptimo para el RMSE. Encogido, cruza el 70 menos veces: avisa menos, así que se le escapan
más hipoglucemias y también se equivoca menos cuando avisa. La persistencia no encoge nada —
copia el valor actual— así que avisa de más, caza más y falla más. Medidos los dos con el mismo
70, el modelo sale con menos sensibilidad y menos falsas alarmas, y de ahí **no se puede
concluir cuál es mejor**: son dos puntos de una curva, no dos resultados.

Medido en la corrida de M-1 del 29/10/2026 sobre el simulador, que es lo que motivó este
módulo:

    M-1            sensibilidad 59.2 %   falsas 20.2 %
    persistencia   sensibilidad 67.8 %   falsas 32.1 %

La pregunta que sí se puede responder es **cuánto caza cada uno cuando los dos se equivocan
igual de seguido**, y eso es lo que mide `sensibilidad_a_falsas_igualadas`.

Aviso que este módulo no puede hacer cumplir y el reporte sí: **elegir el umbral mirando las
métricas del pliegue de prueba es entrenar con la prueba.** Por eso `curva` devuelve la curva
entera, para dibujarla y reportarla, y `umbral_que_cumple_las_metas` recibe los datos sobre los
que está permitido elegir —los pliegues de entrenamiento— y no los de prueba. Quien la llame
con los de prueba obtiene un número que no significa nada, y ninguna comprobación de aquí lo
puede notar.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.eval.metrics import (
    META_FALSAS_ALARMAS_MAX,
    META_SENSIBILIDAD_HIPO,
    UMBRAL_HIPO_DEFECTO,
)

# El barrido por omisión. Empieza por debajo del umbral clínico porque un umbral de aviso más
# bajo que 70 también es una decisión legítima —avisar solo de lo que se predice muy bajo, para
# no molestar— y acaba bien por encima porque un modelo que encoge necesita un umbral alto para
# cruzarlo. Paso de 1 mg/dL: el valor se reporta en enteros.
UMBRALES_POR_OMISION = tuple(range(55, 111))


@dataclass(frozen=True)
class Punto:
    """Un punto de operación: con qué umbral se avisó y qué salió de avisar así."""

    umbral: int
    sensibilidad: float
    falsas: float
    n_alertas: int
    n_episodios: int

    @property
    def cumple_las_metas(self) -> bool:
        """Las dos metas a la vez, que es como se aceptan: ≥ 85 % y ≤ 15 %.

        Por separado cada una es fácil y vacía: un umbral de 110 caza casi todo con un 90 % de
        falsas, y uno de 55 casi no se equivoca porque casi no avisa.
        """
        return (
            self.sensibilidad >= META_SENSIBILIDAD_HIPO and self.falsas <= META_FALSAS_ALARMAS_MAX
        )

    def a_texto(self) -> str:
        marca = "  <-- cumple las dos metas" if self.cumple_las_metas else ""
        return (
            f"aviso < {self.umbral:3} mg/dL   sensibilidad {self.sensibilidad:5.1f} %   "
            f"falsas {self.falsas:5.1f} %   ({self.n_alertas} alertas){marca}"
        )


def punto(
    reales: Sequence[float],
    predichos: Sequence[float],
    umbral_de_aviso: int,
    umbral_clinico: int = UMBRAL_HIPO_DEFECTO,
) -> Punto:
    """Qué sale de avisar cuando lo predicho baja de `umbral_de_aviso`.

    `umbral_clinico` **no se mueve**: lo que cuenta como hipoglucemia real es 70 mg/dL y punto.
    Moverlo sería cambiar la definición de la enfermedad para que el modelo la acierte.
    """
    pares = list(zip(reales, predichos, strict=True))
    episodios = [(r, p) for r, p in pares if r < umbral_clinico]
    if not episodios:
        raise ValueError(
            "no hay episodios reales de hipoglucemia: la sensibilidad no se puede medir y un "
            "100 % en esa casilla sería la cifra más engañosa del reporte"
        )
    alertas = [(r, p) for r, p in pares if p < umbral_de_aviso]
    return Punto(
        umbral=umbral_de_aviso,
        sensibilidad=100 * sum(1 for _, p in episodios if p < umbral_de_aviso) / len(episodios),
        falsas=(
            100 * sum(1 for r, _ in alertas if r >= umbral_clinico) / len(alertas)
            if alertas
            else 0.0
        ),
        n_alertas=len(alertas),
        n_episodios=len(episodios),
    )


def curva(
    reales: Sequence[float],
    predichos: Sequence[float],
    umbrales: Sequence[int] = UMBRALES_POR_OMISION,
    umbral_clinico: int = UMBRAL_HIPO_DEFECTO,
) -> list[Punto]:
    """La curva entera, de umbral bajo a umbral alto. Es lo que va dibujado al reporte.

    Se devuelve completa y no resumida a un número a propósito: el único resumen honesto de un
    compromiso entre cazar y molestar es el compromiso entero. Quien lee el reporte decide
    dónde quiere estar, y en una app de hipoglucemia eso es una decisión de producto, no una
    métrica.
    """
    return [punto(reales, predichos, u, umbral_clinico) for u in sorted(umbrales)]


def sensibilidad_a_falsas_igualadas(
    reales: Sequence[float],
    predichos: Sequence[float],
    falsas_objetivo: float,
    umbrales: Sequence[int] = UMBRALES_POR_OMISION,
    umbral_clinico: int = UMBRAL_HIPO_DEFECTO,
) -> Punto | None:
    """El punto que más caza sin pasarse de `falsas_objetivo`. `None` si ninguno cabe.

    Es la comparación que vale entre dos modelos: igualar lo seguido que molestan y ver quién
    caza más. Comparar sensibilidades a umbrales distintos es comparar dos cosas distintas.

    Devuelve `None` en lugar del punto menos malo cuando ni el umbral más bajo baja de la tasa
    pedida, porque un modelo que se equivoca más de lo aceptable en todos sus puntos de
    operación no tiene un punto de operación aceptable, y ofrecer uno lo escondería.
    """
    caben = [
        p for p in curva(reales, predichos, umbrales, umbral_clinico) if p.falsas <= falsas_objetivo
    ]
    if not caben:
        return None
    return max(caben, key=lambda p: p.sensibilidad)


def umbral_que_cumple_las_metas(
    reales: Sequence[float],
    predichos: Sequence[float],
    umbrales: Sequence[int] = UMBRALES_POR_OMISION,
    umbral_clinico: int = UMBRAL_HIPO_DEFECTO,
) -> Punto | None:
    """El umbral más bajo que cumple las dos metas, o `None` si ninguno las cumple.

    **Los datos que se le pasan tienen que ser los de entrenamiento, nunca los de prueba.**
    Elegir el umbral mirando la prueba y reportar después la métrica de esa misma prueba es
    entrenar con la prueba, y desde aquí no hay forma de notarlo: los números se ven iguales.

    El más bajo de los que cumplen y no el que más caza: subir el umbral compra sensibilidad
    pagando falsas alarmas, y las falsas alarmas son criterio de aceptación porque una app que
    avisa de más se apaga sola. Con las metas ya cumplidas, lo que queda por optimizar es no
    molestar.
    """
    cumplen = [p for p in curva(reales, predichos, umbrales, umbral_clinico) if p.cumple_las_metas]
    return min(cumplen, key=lambda p: p.umbral) if cumplen else None


def comparar(
    reales: Sequence[float],
    predichos_modelo: Sequence[float],
    predichos_base: Sequence[float],
    umbrales: Sequence[int] = UMBRALES_POR_OMISION,
    umbral_clinico: int = UMBRAL_HIPO_DEFECTO,
) -> str:
    """El texto que responde «¿le gana de verdad a la persistencia en hipoglucemias?».

    Iguala la tasa de falsas alarmas de la línea base en su umbral clínico y mide cuánto caza
    el modelo en ese mismo nivel de molestia.
    """
    base = punto(reales, predichos_base, umbral_clinico, umbral_clinico)
    igualado = sensibilidad_a_falsas_igualadas(
        reales, predichos_modelo, base.falsas, umbrales, umbral_clinico
    )
    lineas = [
        f"línea base con aviso < {umbral_clinico} mg/dL: "
        f"sensibilidad {base.sensibilidad:.1f} %, falsas {base.falsas:.1f} %",
    ]
    if igualado is None:
        lineas.append(
            f"el modelo no tiene ningún umbral con falsas <= {base.falsas:.1f} %: en todos sus "
            "puntos de operación molesta más que la línea base"
        )
        return "\n".join(lineas)
    lineas.append(
        f"modelo igualando esas falsas (aviso < {igualado.umbral} mg/dL): "
        f"sensibilidad {igualado.sensibilidad:.1f} %, falsas {igualado.falsas:.1f} %"
    )
    diferencia = igualado.sensibilidad - base.sensibilidad
    if diferencia > 0:
        lineas.append(
            f"a igual molestia el modelo caza {diferencia:.1f} puntos MÁS de hipoglucemias"
        )
    elif diferencia == 0:
        lineas.append(
            "a igual molestia los dos cazan lo mismo: el modelo no aporta nada en "
            "hipoglucemias, aunque su RMSE sea mejor"
        )
    else:
        lineas.append(
            f"a igual molestia el modelo caza {abs(diferencia):.1f} puntos MENOS: el problema "
            "no es el umbral, son las características"
        )
    return "\n".join(lineas)
