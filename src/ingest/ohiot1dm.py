"""Lectura de los XML de OhioT1DM a la forma que usa el resto del repositorio.

## Advertencia que hay que leer antes de usar esto

**El esquema exacto de los XML no está publicado y el conjunto todavía no ha llegado** (la
solicitud a UNC Charlotte sigue pendiente al 29/10/2026). El artículo del conjunto
—Marling y Bunescu, 2020— describe cada campo **en prosa y con espacios en lugar de guiones
bajos** («glucose level», «finger stick»), y no da el nombre del elemento raíz, ni los
nombres de los atributos, ni el formato de las marcas de tiempo.

Así que `SECCIONES` y `FORMATOS_DE_FECHA` de abajo son **la mejor reconstrucción disponible**,
y están en un solo lugar a propósito: cuando lleguen los archivos de verdad, cuadrar el
esquema es editar esa tabla y nada más. Todo lo que viene después —características, partición,
métricas, entrenamiento— trabaja sobre `Lectura` y `Evento`, que no cambian.

`esquema_encontrado()` dice qué trae un archivo real sin interpretarlo, y es lo primero que
hay que correr contra el primer XML que llegue:

```python
from src.ingest.ohiot1dm import esquema_encontrado
print(esquema_encontrado("data/ohiot1dm/2020/train/559-ws-training.xml"))
```

Si lo que imprime no cuadra con `SECCIONES`, se corrige la tabla. **No se adivina leyendo el
código que la usa.**

## Lo que sí está confirmado del artículo

12 personas con diabetes tipo 1, ocho semanas cada una, un XML de entrenamiento y uno de
prueba por persona (24 archivos). Glucosa del sensor cada 5 minutos. El peso viene como
relleno —99 para todos—, así que **no se usa** (y además no entra al modelo). La intensidad
del ejercicio es subjetiva de 1 a 10, la calidad del sueño es 1 pobre, 2 regular, 3 buena, y
las temperaturas vienen en grados Fahrenheit. Las escalas **no se cambian**: son las de
OhioT1DM y por eso los campos de la app se diseñaron así; traducirlas introduce error.
"""

import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Las secciones del archivo y el atributo del que sale su valor. La llave es el nombre del
# elemento que agrupa los eventos; el valor, el atributo numérico de cada `<event>`.
#
# `None` significa que la sección no trae un valor numérico sino solo una marca de tiempo
# (estrés, enfermedad, un episodio de hipoglucemia declarado): el hecho es que ocurrió.
SECCIONES: dict[str, str | None] = {
    "glucose_level": "value",
    "finger_stick": "value",
    "basal": "value",
    "temp_basal": "value",
    "bolus": "dose",
    "meal": "carbs",
    "sleep": "quality",
    "work": "intensity",
    "stressors": None,
    "hypo_event": None,
    "illness": None,
    "exercise": "intensity",
    "basis_heart_rate": "value",
    "basis_gsr": "value",
    "basis_skin_temperature": "value",
    "basis_air_temperature": "value",
    "basis_steps": "value",
    "basis_sleep": "quality",
    "acceleration": "value",
}

# El atributo de la marca de tiempo, y los dos de las secciones que abarcan un periodo
# (un bolo extendido, un turno de trabajo, un tramo de sueño).
TS = "ts"
TS_INICIO = "ts_begin"
TS_FIN = "ts_end"

# Formatos de fecha que se prueban, en orden. El del conjunto es el primero; los demás están
# por si un archivo de un año distinto usa otro. Se prueban en orden y el primero que entra
# gana, así que **no se pueden reordenar sin pensarlo**: `%d-%m-%Y` y `%m-%d-%Y` aceptan los
# dos la misma cadena cuando el día es 12 o menos, y elegir mal desplaza medio conjunto de
# fecha sin que nada falle.
FORMATOS_DE_FECHA: tuple[str, ...] = (
    "%d-%m-%Y %H:%M:%S",
    "%d-%m-%Y %H:%M",
)

# El atributo del sujeto en el elemento raíz. Si el archivo no lo trae, se usa el nombre del
# archivo, que en el conjunto empieza con el id de la persona (por ejemplo `559-ws-training`).
ATRIBUTO_SUJETO = "id"


@dataclass(frozen=True)
class Lectura:
    """Una lectura de glucosa, que es lo que se predice.

    `sujeto` y `ts` son lo que necesita `src/eval/particion`: de quién es la fila y de cuándo.
    `del_sensor` distingue el sensor continuo de un pinchazo, y **el juego denso solo se puede
    construir sobre el sensor**: tres pinchazos sueltos no son una serie, y sus rezagos y
    derivadas serían basura con apariencia de característica.
    """

    sujeto: str
    ts: datetime
    mg_dl: float
    del_sensor: bool

    @property
    def orden(self) -> float:
        """Lo que `particion` usa para ordenar en el tiempo."""
        return self.ts.timestamp()


@dataclass(frozen=True)
class Evento:
    """Cualquier otra cosa con marca de tiempo: una comida, una dosis, ejercicio, sueño.

    `valor` es `None` cuando la sección solo registra que algo pasó. `fin` es `None` salvo en
    las secciones que abarcan un periodo.
    """

    sujeto: str
    seccion: str
    ts: datetime
    valor: float | None = None
    fin: datetime | None = None
    atributos: dict[str, str] = field(default_factory=dict)

    @property
    def orden(self) -> float:
        return self.ts.timestamp()


@dataclass(frozen=True)
class Registro:
    """Todo lo de una persona: sus lecturas y sus eventos por sección."""

    sujeto: str
    lecturas: tuple[Lectura, ...]
    eventos: dict[str, tuple[Evento, ...]]

    @property
    def secciones_vacias(self) -> tuple[str, ...]:
        """Las secciones que el archivo declaró sin un solo evento.

        Importa para el reporte: el artículo dice que varias señales de la banda existen solo
        en algunos sujetos, y un promedio calculado sobre los que sí la traen no se puede
        presentar como si fuera de los doce.
        """
        return tuple(sorted(s for s, e in self.eventos.items() if not e))


def leer_fecha(texto: str) -> datetime:
    """La marca de tiempo de un evento, probando los formatos conocidos en orden.

    Levanta `ValueError` con el texto que no pudo leer, en lugar de devolver `None` o la época
    cero. Una fecha que falla en silencio manda la fila al 1 de enero de 1970 y la deja fuera
    de todas las ventanas sin que nada falle.
    """
    for formato in FORMATOS_DE_FECHA:
        try:
            return datetime.strptime(texto, formato)
        except ValueError:
            continue
    raise ValueError(
        f"no pude leer la fecha {texto!r}. Los formatos que se prueban están en "
        f"FORMATOS_DE_FECHA: {FORMATOS_DE_FECHA}"
    )


def esquema_encontrado(ruta: str | Path) -> dict[str, object]:
    """Qué trae un archivo de verdad, **sin interpretarlo**: el primer paso cuando lleguen.

    Devuelve el nombre y los atributos del elemento raíz, y por cada sección cuántos eventos
    tiene y qué atributos trae el primero. Es deliberadamente tonto: no compara con
    `SECCIONES` ni se queja, solo informa, para poder cuadrar la tabla con lo que hay.
    """
    raiz = ET.parse(str(ruta)).getroot()
    secciones: dict[str, dict[str, object]] = {}
    for hijo in raiz:
        eventos = list(hijo)
        secciones[hijo.tag] = {
            "eventos": len(eventos),
            "atributos_del_primero": sorted(eventos[0].attrib) if eventos else [],
            "ejemplo": dict(eventos[0].attrib) if eventos else {},
        }
    return {
        "raiz": raiz.tag,
        "atributos_de_la_raiz": dict(raiz.attrib),
        "secciones": secciones,
        "secciones_no_esperadas": sorted(set(secciones) - set(SECCIONES)),
        "secciones_esperadas_que_faltan": sorted(set(SECCIONES) - set(secciones)),
    }


def _sujeto_de(raiz: ET.Element, ruta: Path) -> str:
    """El id de la persona: del atributo de la raíz si está, del nombre del archivo si no.

    El nombre del archivo es el respaldo porque en el conjunto empieza con el id
    (`559-ws-training.xml`), y un sujeto mal identificado rompe la partición por persona, que
    es la regla que sostiene todo el protocolo de validación.
    """
    del_atributo = raiz.get(ATRIBUTO_SUJETO)
    if del_atributo:
        return del_atributo
    primer_trozo = ruta.stem.split("-")[0]
    if not primer_trozo:
        raise ValueError(
            f"el archivo {ruta.name} no trae atributo {ATRIBUTO_SUJETO!r} en la raíz y su "
            "nombre no empieza con el id del sujeto"
        )
    return primer_trozo


def _valor(evento: ET.Element, atributo: str | None) -> float | None:
    """El valor numérico de un evento, o `None` si la sección no tiene uno o viene vacío.

    Un valor que no es número se devuelve como `None` en lugar de reventar: el artículo avisa
    de que algunos campos vienen vacíos, y una celda vacía es un dato que falta, no un
    archivo corrupto. Lo que sí revienta es una fecha ilegible, porque sin fecha la fila no se
    puede colocar en ninguna parte.
    """
    if atributo is None:
        return None
    crudo = evento.get(atributo)
    if crudo is None or not crudo.strip():
        return None
    try:
        return float(crudo)
    except ValueError:
        return None


def leer_archivo(ruta: str | Path) -> Registro:
    """Un XML de OhioT1DM a un `Registro`.

    Las secciones que el archivo trae y `SECCIONES` no conoce **se ignoran con su nombre
    anotado** en lugar de reventar: un conjunto que gana un campo en una versión nueva no
    tiene por qué tumbar la lectura. Para saber qué se ignoró está `esquema_encontrado`.
    """
    camino = Path(ruta)
    raiz = ET.parse(str(camino)).getroot()
    sujeto = _sujeto_de(raiz, camino)

    lecturas: list[Lectura] = []
    eventos: dict[str, list[Evento]] = {seccion: [] for seccion in SECCIONES}

    for hijo in raiz:
        if hijo.tag not in SECCIONES:
            continue
        atributo = SECCIONES[hijo.tag]
        for evento in hijo:
            crudo_ts = evento.get(TS) or evento.get(TS_INICIO)
            if crudo_ts is None:
                continue
            ts = leer_fecha(crudo_ts)
            valor = _valor(evento, atributo)

            if hijo.tag in ("glucose_level", "finger_stick"):
                if valor is None:
                    continue
                lecturas.append(
                    Lectura(
                        sujeto=sujeto,
                        ts=ts,
                        mg_dl=valor,
                        del_sensor=hijo.tag == "glucose_level",
                    )
                )
                continue

            crudo_fin = evento.get(TS_FIN)
            eventos[hijo.tag].append(
                Evento(
                    sujeto=sujeto,
                    seccion=hijo.tag,
                    ts=ts,
                    valor=valor,
                    fin=leer_fecha(crudo_fin) if crudo_fin else None,
                    atributos=dict(evento.attrib),
                )
            )

    # Ordenadas por tiempo aquí, una sola vez. El resto del repositorio da por hecho que lo
    # están, y reordenar en cada uso sería lento y fácil de olvidar en un sitio.
    return Registro(
        sujeto=sujeto,
        lecturas=tuple(sorted(lecturas, key=lambda f: f.ts)),
        eventos={s: tuple(sorted(e, key=lambda x: x.ts)) for s, e in eventos.items()},
    )


def leer_carpeta(carpeta: str | Path) -> Iterator[Registro]:
    """Todos los XML de una carpeta, en orden de nombre.

    En orden y no como los dé el sistema de archivos: dos corridas que leen en distinto orden
    pueden partir distinto, y un resultado que no se puede repetir no sirve para un trabajo
    que se va a evaluar.
    """
    for ruta in sorted(Path(carpeta).glob("*.xml")):
        yield leer_archivo(ruta)
