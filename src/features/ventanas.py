"""De una serie de lecturas a las filas que entrenan, con su objetivo.

Es la pieza donde se cometen los errores que no se ven. Tres, y las tres producen métricas
buenísimas y mentirosas:

1. **Mirar hacia adelante.** Una característica que use una lectura posterior al instante que
   se está prediciendo. En producción no existe: a las 14:00 el teléfono no tiene las 14:05.
   Aquí cada fila se arma con `serie[:i+1]` y nada más, y hay una prueba que lo comprueba
   recortando la serie y exigiendo el mismo vector.
2. **Rellenar lo que falta con cero.** Un rezago de 60 minutos que no existe porque la serie
   empieza ahí no es una glucosa de cero. Va `NaN`, que XGBoost entiende.
3. **Suponer una serie regular de cinco minutos.** HealthKit entrega lo que la app del
   fabricante escribió, cuando lo escribió, y OhioT1DM trae huecos. Así que **los rezagos se
   buscan por tiempo y no por posición**: `rezago_30` es «la lectura más cercana a hace 30
   minutos, si existe dentro de la tolerancia», no «la fila de cinco atrás».

Y una regla que no es técnica: **el objetivo también se busca por tiempo**. Si no hay lectura
cerca de los 30 minutos, esa fila **no se usa** en lugar de usar la más próxima que haya; un
objetivo a 52 minutos etiquetado como «30 minutos» es ruido con apariencia de dato.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from src.features.features import (
    CARACTERISTICAS_DENSAS,
    CARACTERISTICAS_DISPERSAS,
    FALTA,
    REZAGOS_MIN,
    codigo_contexto,
    hora_ciclica,
)
from src.ingest.ohiot1dm import Evento, Lectura

# Cuánto puede alejarse una lectura del instante que se le pide para seguir sirviendo. La mitad
# del intervalo nominal del sensor: con 2.5 min, cada instante tiene a lo más un candidato y no
# hay que decidir empates.
TOLERANCIA_REZAGO_MIN = 2.5

# La misma tolerancia para el objetivo. Más estrecha sería tirar filas por nada; más amplia
# etiquetaría como «a 30 minutos» una lectura que está a 40.
TOLERANCIA_OBJETIVO_MIN = 5.0

# Minutos que se consideran para los promedios móviles.
VENTANAS_MEDIA_MIN = (30, 60)

SEGUNDOS_POR_MINUTO = 60


@dataclass(frozen=True)
class Fila:
    """Una fila de entrenamiento: de quién es, de cuándo, qué entra y qué hay que predecir.

    `sujeto` y `orden` son lo que `src/eval/particion` necesita, así que una `Fila` se puede
    repartir sin convertirla.
    """

    sujeto: str
    ts_segundos: float
    caracteristicas: dict[str, float]
    objetivo: float

    @property
    def orden(self) -> float:
        return self.ts_segundos


def _minutos(a: float, b: float) -> float:
    """Minutos de `b` a `a`, con `a` y `b` en segundos."""
    return (a - b) / SEGUNDOS_POR_MINUTO


def _mas_cercana(
    lecturas: Sequence[Lectura], objetivo_seg: float, tolerancia_min: float
) -> Lectura | None:
    """La lectura más cercana a un instante, o `None` si ninguna cae dentro de la tolerancia.

    Búsqueda lineal a propósito: con ocho semanas por sujeto el conjunto entra en memoria y una
    búsqueda binaria aquí ahorraría segundos a cambio de un error de índice que nadie vería.
    Si algún día estorba, se cambia con las pruebas puestas.
    """
    mejor: Lectura | None = None
    mejor_distancia = math.inf
    for lectura in lecturas:
        distancia = abs(_minutos(lectura.ts.timestamp(), objetivo_seg))
        if distancia < mejor_distancia:
            mejor, mejor_distancia = lectura, distancia
    if mejor is None or mejor_distancia > tolerancia_min:
        return None
    return mejor


def cob(comidas: Sequence[Evento], ahora_seg: float, absorcion_min: float = 180) -> float:
    """Carbohidratos que siguen activos, con absorción lineal: `carbs * (1 - t/absorcion)`.

    Lineal y no una curva de digestión: es la decisión D-8 del proyecto, y los mismos números
    los da `referencia/reglas_clinicas.py` del backend. El caso que se cita siempre: 60 g con
    absorción de 180 min, a los 90 minutos, son 30 g.

    **Solo cuenta lo que la persona registró.** Una comida que no está no se infiere (D-9,
    RF-37), así que una subida sin comida registrada entra al modelo como una subida sin COB.
    """
    total = 0.0
    for comida in comidas:
        if comida.valor is None:
            continue
        transcurrido = _minutos(ahora_seg, comida.ts.timestamp())
        if transcurrido < 0 or transcurrido >= absorcion_min:
            continue
        total += comida.valor * (1 - transcurrido / absorcion_min)
    return total


def iob(dosis: Sequence[Evento], ahora_seg: float, duracion_h: float = 5.0) -> float:
    """Insulina que sigue activa, con la curva del perfil: pico alrededor de 75 min.

    La fracción activa se aproxima con la misma forma que usa el backend: sube hasta el pico y
    baja hasta cero al terminar la duración de acción. Aquí **no se calcula ninguna dosis**
    (regla 7 del proyecto): se contabiliza lo que la persona dijo haberse aplicado.
    """
    duracion_min = duracion_h * 60
    pico_min = 75.0
    total = 0.0
    for d in dosis:
        if d.valor is None:
            continue
        t = _minutos(ahora_seg, d.ts.timestamp())
        if t < 0 or t >= duracion_min:
            continue
        if t <= pico_min:
            # Antes del pico casi nada se ha consumido: la fracción activa baja despacio.
            fraccion = 1 - 0.25 * (t / pico_min)
        else:
            fraccion = 0.75 * (1 - (t - pico_min) / (duracion_min - pico_min))
        total += d.valor * max(fraccion, 0.0)
    return total


def _rezagos(historia: Sequence[Lectura], ahora_seg: float) -> dict[str, float]:
    """Los cinco rezagos, buscados **por tiempo**. Lo que no está, `NaN`."""
    valores: dict[str, float] = {}
    for minutos in REZAGOS_MIN:
        encontrada = _mas_cercana(
            historia, ahora_seg - minutos * SEGUNDOS_POR_MINUTO, TOLERANCIA_REZAGO_MIN
        )
        valores[f"rezago_{minutos}"] = encontrada.mg_dl if encontrada else FALTA
    return valores


def _derivadas(actual: float, rezagos: dict[str, float]) -> dict[str, float]:
    """Primera y segunda derivada en mg/dL por minuto, a partir de los rezagos de 5 y 10.

    `NaN` cuando falta el rezago que hace falta, en lugar de suponer cero cambio: una serie que
    empieza no es una serie plana, y un modelo que vea «cambio cero» al arrancar aprenderá a
    predecir quietud justo cuando no sabe nada.
    """
    r5, r10 = rezagos["rezago_5"], rezagos["rezago_10"]
    d1 = (actual - r5) / 5 if not math.isnan(r5) else FALTA
    if math.isnan(r5) or math.isnan(r10):
        return {"derivada_1": d1, "derivada_2": FALTA}
    d1_anterior = (r5 - r10) / 5
    return {"derivada_1": d1, "derivada_2": (d1 - d1_anterior) / 5}


def _medias(historia: Sequence[Lectura], ahora_seg: float) -> dict[str, float]:
    """Promedios móviles de 30 y 60 minutos. Sin lecturas en la ventana, `NaN`."""
    valores: dict[str, float] = {}
    for minutos in VENTANAS_MEDIA_MIN:
        desde = ahora_seg - minutos * SEGUNDOS_POR_MINUTO
        dentro = [f.mg_dl for f in historia if desde <= f.ts.timestamp() <= ahora_seg]
        valores[f"media_{minutos}"] = sum(dentro) / len(dentro) if dentro else FALTA
    return valores


def caracteristicas_densas(
    historia: Sequence[Lectura],
    comidas: Sequence[Evento],
    dosis: Sequence[Evento],
    *,
    absorcion_min: float = 180,
    duracion_insulina_h: float = 5.0,
) -> dict[str, float]:
    """El juego denso para el último punto de `historia`. **Solo mira hacia atrás.**

    `historia` tiene que terminar en el instante que se predice y venir ordenada. Es
    responsabilidad de quien llama, y `filas_densas` lo garantiza pasando `serie[:i+1]`.
    """
    if not historia:
        raise ValueError("no hay historia con la que armar la fila")
    ahora = historia[-1]
    ahora_seg = ahora.ts.timestamp()
    rezagos = _rezagos(historia, ahora_seg)
    sin, cos = hora_ciclica(ahora.ts.hour * 60 + ahora.ts.minute)

    fila: dict[str, float] = {
        "mg_dl": ahora.mg_dl,
        **rezagos,
        **_derivadas(ahora.mg_dl, rezagos),
        **_medias(historia, ahora_seg),
        "cob": cob(comidas, ahora_seg, absorcion_min),
        "iob": iob(dosis, ahora_seg, duracion_insulina_h),
        "hora_sin": sin,
        "hora_cos": cos,
        "dia_semana": float(ahora.ts.weekday()),
    }
    faltan = set(CARACTERISTICAS_DENSAS) - set(fila)
    sobran = set(fila) - set(CARACTERISTICAS_DENSAS)
    if faltan or sobran:
        raise AssertionError(
            f"el juego denso no cuadra con la tupla del contrato: faltan {sorted(faltan)}, "
            f"sobran {sorted(sobran)}"
        )
    return fila


def caracteristicas_dispersas(
    ultima: Lectura,
    ahora_seg: float,
    comidas: Sequence[Evento],
    dosis: Sequence[Evento],
    *,
    promedio_7d: float = FALTA,
    contexto: str | None = None,
    absorcion_min: float = 180,
    duracion_insulina_h: float = 5.0,
) -> dict[str, float]:
    """El juego disperso: una sola lectura y su antigüedad, sin serie.

    Es el modo sin sensor, y por eso **no lleva ni un rezago ni una derivada**: tres pinchazos
    separados por horas no son una serie, y proyectar con ellos sería dar apariencia de
    predicción a un dato que no la sostiene. `antiguedad_min` es la variable que carga con eso,
    y es la que el modelo usa para saber cuánto desconfiar.
    """
    ultima_comida = max(comidas, key=lambda c: c.ts, default=None)
    ultima_dosis = max(dosis, key=lambda d: d.ts, default=None)
    sin, cos = hora_ciclica((ahora_seg % 86400) / SEGUNDOS_POR_MINUTO)

    fila: dict[str, float] = {
        "mg_dl": ultima.mg_dl,
        "antiguedad_min": _minutos(ahora_seg, ultima.ts.timestamp()),
        "carbs_g": (
            ultima_comida.valor
            if ultima_comida is not None and ultima_comida.valor is not None
            else FALTA
        ),
        "dosis_ui": (
            ultima_dosis.valor
            if ultima_dosis is not None and ultima_dosis.valor is not None
            else FALTA
        ),
        "cob": cob(comidas, ahora_seg, absorcion_min),
        "iob": iob(dosis, ahora_seg, duracion_insulina_h),
        "hora_sin": sin,
        "hora_cos": cos,
        "contexto_comida": codigo_contexto(contexto),
        "promedio_7d": promedio_7d,
    }
    faltan = set(CARACTERISTICAS_DISPERSAS) - set(fila)
    sobran = set(fila) - set(CARACTERISTICAS_DISPERSAS)
    if faltan or sobran:
        raise AssertionError(
            f"el juego disperso no cuadra con la tupla del contrato: faltan {sorted(faltan)}, "
            f"sobran {sorted(sobran)}"
        )
    return fila


def filas_densas(
    serie: Sequence[Lectura],
    comidas: Sequence[Evento],
    dosis: Sequence[Evento],
    *,
    horizonte_min: int,
    solo_sensor: bool = True,
    **opciones: float,
) -> list[Fila]:
    """Todas las filas de entrenamiento de una serie, con su objetivo a `horizonte_min`.

    `solo_sensor` deja fuera los pinchazos por omisión: el juego denso se construye sobre la
    serie continua, y mezclar puntos aislados metería rezagos falsos.

    **Una fila sin objetivo dentro de la tolerancia se descarta.** Es la decisión que más filas
    tira en un conjunto con huecos, y la correcta: usar la lectura más próxima que haya
    etiquetaría como «a 30 minutos» un valor que está a 50.
    """
    if horizonte_min <= 0:
        raise ValueError("el horizonte debe ser de al menos un minuto")
    usables = [f for f in serie if f.del_sensor] if solo_sensor else list(serie)
    if any(
        b.ts < a.ts for a, b in zip(usables, usables[1:], strict=False)
    ):  # pragma: no cover - lo garantiza el lector
        raise ValueError("la serie tiene que venir ordenada en el tiempo")

    filas: list[Fila] = []
    for i, ahora in enumerate(usables):
        ahora_seg = ahora.ts.timestamp()
        objetivo = _mas_cercana(
            usables[i + 1 :],
            ahora_seg + horizonte_min * SEGUNDOS_POR_MINUTO,
            TOLERANCIA_OBJETIVO_MIN,
        )
        if objetivo is None:
            continue
        # `usables[: i + 1]` es la garantía de no mirar adelante: la fila se arma con la
        # historia hasta ahora inclusive y con nada más.
        filas.append(
            Fila(
                sujeto=ahora.sujeto,
                ts_segundos=ahora_seg,
                caracteristicas=caracteristicas_densas(
                    usables[: i + 1],
                    [c for c in comidas if c.ts.timestamp() <= ahora_seg],
                    [d for d in dosis if d.ts.timestamp() <= ahora_seg],
                    **opciones,  # type: ignore[arg-type]
                ),
                objetivo=objetivo.mg_dl,
            )
        )
    return filas


@dataclass(frozen=True)
class FilaPico:
    """Una comida y lo que pasó después: el máximo de glucosa y a los cuántos minutos.

    Es lo que predice M-3 (RF-27), y son **dos objetivos**, no uno: «vas a llegar a 210» y
    «vas a llegar en 70 minutos» son dos preguntas, y una sola no sirve. Se guardan juntos
    porque salen de la misma ventana y separarlos invitaría a desalinearlos.
    """

    sujeto: str
    ts_segundos: float
    caracteristicas: dict[str, float]
    pico_mg_dl: float
    pico_min: float

    @property
    def orden(self) -> float:
        return self.ts_segundos


def filas_pico(
    serie: Sequence[Lectura],
    comidas: Sequence[Evento],
    dosis: Sequence[Evento],
    *,
    ventana_min: int = 180,
    minimo_lecturas: int = 6,
    **opciones: float,
) -> list[FilaPico]:
    """Una fila por comida registrada, con el pico de las siguientes `ventana_min`.

    Tres decisiones que cambian el resultado:

    1. **Una fila por comida registrada, y nada más.** No se buscan picos sin comida detrás:
       una subida sin comida registrada no se atribuye a una comida que la persona no anotó
       (D-9, RF-37). Esas subidas existen en los datos y se quedan fuera de M-3 a propósito.
    2. **Las características son las del juego disperso**, porque M-3 es el modelo del modo sin
       sensor: tiene la última lectura y su antigüedad, no una serie. Entrenarlo con rezagos lo
       volvería inútil justo para quien no tiene sensor.
    3. **Una ventana con pocas lecturas se descarta** (`minimo_lecturas`). El máximo de dos
       puntos sueltos no es un pico posprandial, y etiquetarlo como tal mete ruido con
       apariencia de dato en el único modelo que mira las comidas.

    La ventana por omisión son 180 minutos, que es el tiempo de absorción que usa el proyecto
    (D-8): pasado eso, el COB es cero y lo que suba ya no es de esa comida.
    """
    if ventana_min <= 0:
        raise ValueError("la ventana tiene que ser de al menos un minuto")

    filas: list[FilaPico] = []
    for comida in comidas:
        if comida.valor is None:
            continue
        comida_seg = comida.ts.timestamp()
        previas = [f for f in serie if f.ts.timestamp() <= comida_seg]
        if not previas:
            continue
        hasta = comida_seg + ventana_min * SEGUNDOS_POR_MINUTO
        dentro = [f for f in serie if comida_seg < f.ts.timestamp() <= hasta]
        if len(dentro) < minimo_lecturas:
            continue
        pico = max(dentro, key=lambda f: f.mg_dl)
        filas.append(
            FilaPico(
                sujeto=comida.sujeto,
                ts_segundos=comida_seg,
                caracteristicas=caracteristicas_dispersas(
                    previas[-1],
                    comida_seg,
                    [c for c in comidas if c.ts.timestamp() <= comida_seg],
                    [d for d in dosis if d.ts.timestamp() <= comida_seg],
                    promedio_7d=_promedio_previo(previas, comida_seg),
                    **opciones,  # type: ignore[arg-type]
                ),
                pico_mg_dl=pico.mg_dl,
                pico_min=_minutos(pico.ts.timestamp(), comida_seg),
            )
        )
    return filas


def _promedio_previo(previas: Sequence[Lectura], ahora_seg: float, dias: int = 7) -> float:
    """El promedio de los últimos `dias` días, mirando solo hacia atrás.

    Es la característica `promedio_7d` del juego disperso, y es la que le da al modelo sin
    sensor una idea de en qué nivel vive esa persona. `NaN` si no hay nada en la ventana, como
    todo lo que falta.
    """
    desde = ahora_seg - dias * 24 * 3600
    dentro = [f.mg_dl for f in previas if desde <= f.ts.timestamp() <= ahora_seg]
    return sum(dentro) / len(dentro) if dentro else FALTA
