"""Series de juguete para poder probar el entrenamiento **sin** el conjunto de datos.

## Advertencia que importa más que el código

**Esto no es un simulador fisiológico y no sirve para entrenar un modelo que se vaya a
reportar.** Es un generador deliberadamente simple cuyo único propósito es que el resto del
repositorio se pueda ejecutar y probar antes de que llegue OhioT1DM: que un `train_m1.py`
falle hoy por un error de programación y no el día que lleguen los datos.

Si el acceso a OhioT1DM no llega, el respaldo del proyecto es un **simulador fisiológico de
código abierto**, con menos validez y dicho como tal en el reporte. **No es este archivo.**
Una métrica calculada sobre estas series no significa nada: la señal la generó una suma de
senos y por construcción es predecible, así que cualquier modelo decente saca un RMSE
excelente. Por eso `src/train` escribe en el manifiesto del artefacto **de dónde salieron los
datos**, y un artefacto entrenado con esto queda marcado como `sintetico`.
"""

import math
import random
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta

from src.ingest.ohiot1dm import Evento, Lectura, Registro

ORIGEN = "sintetico"

# Los valores alrededor de los que se mueve la señal. Están calibrados para que la mezcla de
# las doce personas se parezca a la de un conjunto real. Con doce personas y tres días sale
# alrededor de **74 % del tiempo en rango, 1.7 % por debajo de 70 y picos de 278 mg/dL**, y la
# tasa de hipoglucemias va de 0 a casi 6 % según la persona. Importa porque si la serie nunca
# baja de 70 la sensibilidad a hipoglucemia no se puede probar, y es la métrica que más importa;
# y si nunca pasa de 180, la zona C de la rejilla de Clarke jamás se ejercita.
#
# El nivel de base **varía por persona** (`Perfil.base_mg_dl`): unas corren bajas y otras altas,
# que es lo que hace que repartir por persona dé un resultado distinto de repartir al azar.
BASE_MIN, BASE_MAX = 115.0, 170.0
AMPLITUD_DIARIA = 50.0
AMPLITUD_RAPIDA = 12.0
RUIDO_MG_DL = 4.0
MINIMO, MAXIMO = 40.0, 320.0

INTERVALO_MIN = 5


@dataclass(frozen=True)
class Perfil:
    """Lo que hace distinta a una persona sintética.

    Existe para que **partir por sujeto signifique algo** en las pruebas: si las doce series
    fueran idénticas, un reparto por persona y uno al azar darían lo mismo y las pruebas de
    `particion` no estarían comprobando nada.
    """

    sujeto: str
    base_mg_dl: float
    desfase_h: float
    sensibilidad_carbos: float
    sensibilidad_insulina: float


def perfiles(cuantos: int = 12, semilla: int = 11) -> list[Perfil]:
    """`cuantos` personas distintas, reproducibles a partir de la semilla.

    Doce por omisión, como OhioT1DM, para que las pruebas de LOSO trabajen con el mismo número
    de pliegues que la corrida de verdad.
    """
    rng = random.Random(semilla)
    return [
        Perfil(
            sujeto=f"S{i:02d}",
            base_mg_dl=rng.uniform(BASE_MIN, BASE_MAX),
            desfase_h=rng.uniform(0, 24),
            sensibilidad_carbos=rng.uniform(8.0, 13.0),
            sensibilidad_insulina=rng.uniform(30.0, 65.0),
        )
        for i in range(cuantos)
    ]


def _comidas_y_dosis(
    perfil: Perfil, inicio: datetime, dias: int, rng: random.Random
) -> tuple[list[Evento], list[Evento]]:
    """Tres comidas al día con su dosis, a horas que se mueven un poco.

    Cada comida lleva su dosis porque es lo que hace una persona con diabetes tipo 1, y porque
    así COB e IOB tienen algo que medir. **Aquí no se calcula ninguna dosis terapéutica**: se
    generan números para que el código tenga qué procesar.
    """
    comidas: list[Evento] = []
    dosis: list[Evento] = []
    for dia in range(dias):
        for hora_nominal, carbos in ((8, 45.0), (14, 70.0), (20, 55.0)):
            cuando = inicio + timedelta(days=dia, hours=hora_nominal + rng.uniform(-0.75, 0.75))
            gramos = round(carbos * rng.uniform(0.7, 1.3), 1)
            comidas.append(Evento(sujeto=perfil.sujeto, seccion="meal", ts=cuando, valor=gramos))
            dosis.append(
                Evento(
                    sujeto=perfil.sujeto,
                    seccion="bolus",
                    ts=cuando + timedelta(minutes=rng.uniform(-10, 5)),
                    valor=round(gramos / 10 * rng.uniform(0.8, 1.2), 1),
                )
            )
    return sorted(comidas, key=lambda e: e.ts), sorted(dosis, key=lambda e: e.ts)


def _valor(
    perfil: Perfil,
    cuando: datetime,
    comidas: list[Evento],
    dosis: list[Evento],
    rng: random.Random,
) -> float:
    """Un valor de glucosa: ritmo diario, oscilación rápida, efecto de comidas y de insulina.

    La forma importa poco; lo que importa es que **las comidas suban la señal y la insulina la
    baje**, porque si no, `cob` e `iob` entrarían al modelo como ruido y las pruebas no podrían
    distinguir un entrenamiento que funciona de uno que no.
    """
    from src.features.ventanas import cob, iob  # import local: evita un ciclo de módulos

    ahora_seg = cuando.timestamp()
    horas = ahora_seg / 3600 + perfil.desfase_h
    senal = (
        perfil.base_mg_dl
        + AMPLITUD_DIARIA * math.sin(2 * math.pi * horas / 24)
        + AMPLITUD_RAPIDA * math.sin(2 * math.pi * horas / 3.5)
        + perfil.sensibilidad_carbos * cob(comidas, ahora_seg) / 10
        - perfil.sensibilidad_insulina * iob(dosis, ahora_seg) / 10
        + rng.gauss(0, RUIDO_MG_DL)
    )
    return round(min(max(senal, MINIMO), MAXIMO), 1)


def registro(
    perfil: Perfil,
    *,
    dias: int = 7,
    inicio: datetime | None = None,
    semilla: int = 7,
    probabilidad_de_hueco: float = 0.02,
) -> Registro:
    """Un `Registro` como el que devolvería el lector de OhioT1DM, con la misma forma.

    `probabilidad_de_hueco` tira lecturas al azar a propósito: **el conjunto real tiene
    huecos**, HealthKit tampoco garantiza una serie regular, y un generador perfecto esconde
    justo los errores de rezagos por posición que `ventanas.py` existe para evitar.
    """
    rng = random.Random(f"{semilla}-{perfil.sujeto}")
    arranque = inicio or datetime(2026, 1, 5, 0, 0)
    comidas, dosis = _comidas_y_dosis(perfil, arranque, dias, rng)

    lecturas: list[Lectura] = []
    for paso in range(dias * 24 * 60 // INTERVALO_MIN):
        if rng.random() < probabilidad_de_hueco:
            continue
        cuando = arranque + timedelta(minutes=paso * INTERVALO_MIN)
        lecturas.append(
            Lectura(
                sujeto=perfil.sujeto,
                ts=cuando,
                mg_dl=_valor(perfil, cuando, comidas, dosis, rng),
                del_sensor=True,
            )
        )

    eventos = {"meal": tuple(comidas), "bolus": tuple(dosis)}
    return Registro(sujeto=perfil.sujeto, lecturas=tuple(lecturas), eventos=eventos)


def conjunto(cuantos: int = 12, *, dias: int = 7, semilla: int = 7) -> Iterator[Registro]:
    """Varios registros, uno por persona sintética."""
    for perfil in perfiles(cuantos, semilla=semilla):
        yield registro(perfil, dias=dias, semilla=semilla)
