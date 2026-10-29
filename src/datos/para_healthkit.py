"""Exporta una serie de glucosa al CSV que lee la utilidad de pruebas del iPhone.

## Para qué existe

La fase 5 valida el modo sensor con **48 horas continuas**: huecos medidos, retrasos medidos y
la alerta encendida. Pide un sensor comercial, y el proyecto todavía no tiene uno; la fecha
para decidir si se consigue es el **1 de noviembre de 2026**.

El plan B documentado es escribir una serie en HealthKit desde una utilidad de pruebas, para
que la app la lea como leería la de un Dexcom o un Libre. Este archivo produce lo que esa
utilidad necesita: un CSV de tres columnas.

## Lo que este plan B sí valida y lo que no

Valida **la plomería**, que es donde están las tres trampas documentadas de HealthKit: la
consulta de observación en segundo plano, no reescribir lo que se leyó (RF-15b) y no caerse
cuando la persona revoca el permiso. Y la valida **mejor** que un sensor real en un aspecto:
los huecos y los retrasos se piden a propósito en lugar de esperar a que ocurran.

No valida el comportamiento del sensor comercial: su cadencia real, sus recuperaciones, sus
calibraciones. **Eso hay que decirlo en el reporte (riesgo R-8)**, no dejar creer que se
validó con hardware.

## Por qué los huecos y los retrasos se escriben a propósito

Porque son el caso que rompe las cosas. HealthKit **no entrega una serie regular de cinco
minutos**: entrega lo que la app del fabricante escribió, cuando lo escribió, y puede llegar
en bloques y con retraso (RF-06b). Una utilidad que escribiera una serie perfecta cada cinco
minutos validaría el único escenario que nunca ocurre.

Por eso el CSV lleva dos columnas de tiempo:

- `ts_medicion` — cuándo ocurrió la lectura. Es lo que va en el `HKQuantitySample`.
- `ts_escritura` — cuándo la utilidad debe escribirla en HealthKit.

Cuando las dos coinciden, la escritura es puntual. Cuando `ts_escritura` va después, la
lectura llega tarde, y **todas las lecturas del mismo bloque llevan la misma `ts_escritura`**:
así es como llega un sensor real cuando el teléfono estuvo fuera de alcance y vuelve, de golpe
y no una por una. La utilidad ordena por `ts_escritura` y respeta las pausas.

## La zona horaria

Las marcas de tiempo salen con desplazamiento explícito (`-06:00`), no en hora local suelta: la
regla 5 del proyecto es UTC más la zona, y un archivo que dice «14:05» sin decir de dónde es
inservible en cuanto cruza un huso.

Las series de origen no traen zona —ni las de OhioT1DM, que son hora local del sujeto sin decir
cuál, ni las del generador de juguete—, así que aquí se les **asigna** la de México. Es una
convención del archivo de pruebas, no un dato del conjunto, y por eso se dice aquí.
"""

import argparse
import csv
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# Un hueco de más de 15 minutos ya no se interpola (RF-23), así que los huecos del archivo se
# reparten a los dos lados de ese umbral: hay que ver las dos ramas del código.
HUECOS_CORTOS_MIN = (5, 10, 15)
HUECOS_LARGOS_MIN = (25, 45, 90)

# Cada cuánto llega un bloque atrasado, y cuánto se atrasa.
CADA_CUANTAS_LECTURAS_UN_RETRASO = 150
RETRASO_MIN = 35

CABECERA = ("ts_medicion", "ts_escritura", "mg_dl")

# Las series de origen no traen zona; se les asigna la del proyecto (regla 5).
ZONA = ZoneInfo("America/Mexico_City")


@dataclass(frozen=True)
class Punto:
    """Una lectura con sus dos tiempos: cuándo ocurrió y cuándo se escribe."""

    ts_medicion: datetime
    ts_escritura: datetime
    mg_dl: float

    @property
    def llega_tarde(self) -> bool:
        return self.ts_escritura > self.ts_medicion


def con_huecos_y_retrasos(
    lecturas: Iterable[tuple[datetime, float]],
    *,
    semilla: int = 5,
    con_huecos: bool = True,
    con_retrasos: bool = True,
) -> Iterator[Punto]:
    """La serie, perforada y con bloques que llegan tarde.

    Los huecos se hacen **quitando** lecturas, no poniendo un valor nulo: así es como falta un
    dato en HealthKit, que simplemente no tiene la muestra. Un cero o un `NaN` escrito en la
    base sería otra cosa, y la app tendría que tratarlo distinto.

    La semilla es fija para que dos corridas produzcan el mismo archivo: una prueba de 48 horas
    que no se puede repetir no sirve para comparar un arreglo contra el problema.
    """
    azar = random.Random(semilla)
    bloque_hasta: datetime | None = None
    saltar_hasta: datetime | None = None

    for indice, (ts, mg_dl) in enumerate(lecturas):
        ts = ts if ts.tzinfo is not None else ts.replace(tzinfo=ZONA)

        if saltar_hasta is not None and ts < saltar_hasta:
            continue
        saltar_hasta = None

        if con_retrasos and indice and indice % CADA_CUANTAS_LECTURAS_UN_RETRASO == 0:
            bloque_hasta = ts + timedelta(minutes=RETRASO_MIN)

        if bloque_hasta is not None and ts < bloque_hasta:
            # El teléfono estuvo fuera de alcance: la lectura se midió ahora y se escribirá
            # cuando vuelva, junto con todas las demás del bloque y con la misma marca.
            ts_escritura = bloque_hasta
        else:
            bloque_hasta = None
            ts_escritura = ts

        yield Punto(ts_medicion=ts, ts_escritura=ts_escritura, mg_dl=mg_dl)

        if con_huecos and azar.random() < 0.004:
            largo = azar.choice(HUECOS_CORTOS_MIN + HUECOS_LARGOS_MIN)
            saltar_hasta = ts + timedelta(minutes=largo)


def escribir(ruta: Path | str, puntos: Iterable[Punto]) -> int:
    """Escribe el CSV y devuelve cuántas filas quedaron.

    Devuelve el número de filas en lugar de nada para que quien llama pueda comprobar que
    escribió lo que creía: `puntos` puede ser un generador, y contarlo después de agotarlo da
    cero.
    """
    ruta = Path(ruta)
    filas = 0
    with ruta.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(CABECERA)
        for punto in puntos:
            escritor.writerow(
                [
                    punto.ts_medicion.isoformat(),
                    punto.ts_escritura.isoformat(),
                    f"{punto.mg_dl:.1f}",
                ]
            )
            filas += 1
    return filas


def leer(ruta: Path | str) -> list[Punto]:
    """Vuelve a leer el CSV. Existe para que una prueba compruebe el viaje redondo."""
    with Path(ruta).open(encoding="utf-8", newline="") as archivo:
        lector = csv.reader(archivo)
        cabecera = tuple(next(lector))
        if cabecera != CABECERA:
            raise ValueError(f"cabecera inesperada: {cabecera}, se esperaba {CABECERA}")
        return [
            Punto(
                ts_medicion=datetime.fromisoformat(medicion),
                ts_escritura=datetime.fromisoformat(escritura),
                mg_dl=float(mg_dl),
            )
            for medicion, escritura, mg_dl in lector
        ]


def resumen(puntos: list[Punto]) -> str:
    """Lo que hay que poder decir del archivo en el reporte, sin abrirlo."""
    if not puntos:
        return "el archivo quedó vacío"
    tarde = [p for p in puntos if p.llega_tarde]
    bloques = len({p.ts_escritura for p in tarde})
    ordenados = sorted(p.ts_medicion for p in puntos)
    huecos = [
        (b - a).total_seconds() / 60
        for a, b in zip(ordenados, ordenados[1:], strict=False)
        if (b - a).total_seconds() / 60 > 5
    ]
    horas = (ordenados[-1] - ordenados[0]).total_seconds() / 3600
    return (
        f"{len(puntos)} lecturas en {horas:.1f} h · {len(huecos)} huecos "
        f"({sum(1 for h in huecos if h > 15)} de más de 15 min, el mayor de "
        f"{max(huecos, default=0):.0f} min) · {len(tarde)} lecturas que llegan tarde, "
        f"en {bloques} bloques"
    )


def _lecturas_del_origen(args: argparse.Namespace) -> list[tuple[datetime, float]]:
    """Las lecturas del sensor, de OhioT1DM o del generador de juguete.

    Solo las del sensor continuo: tres pinchazos sueltos no son una serie y escribirlos como
    si lo fueran le haría creer a la app que tiene sensor cuando no lo tiene.
    """
    if args.datos == "ohiot1dm":
        from src.ingest.ohiot1dm import leer_carpeta

        registros = list(leer_carpeta(args.carpeta))
        if not registros:
            raise SystemExit(f"no hay archivos de OhioT1DM en {args.carpeta}")
        elegido = next((r for r in registros if r.sujeto == args.sujeto), registros[0])
    else:
        from src.datos.sintetico import conjunto

        elegido = next(iter(conjunto(1, dias=args.dias + 1, semilla=args.semilla)))

    lecturas = sorted(
        ((lec.ts, lec.mg_dl) for lec in elegido.lecturas if lec.del_sensor),
        key=lambda par: par[0],
    )
    if not lecturas:
        raise SystemExit("el sujeto elegido no trae lecturas del sensor continuo")
    hasta = lecturas[0][0] + timedelta(days=args.dias)
    return [(ts, mg) for ts, mg in lecturas if ts <= hasta]


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Exporta una serie de glucosa al CSV que lee la utilidad de pruebas del iPhone, "
            "para validar el modo sensor sin un sensor comercial (fase 5, riesgo R-8)."
        )
    )
    p.add_argument("--datos", default="sintetico", choices=("ohiot1dm", "sintetico"))
    p.add_argument("--carpeta", default="data", help="carpeta de los XML, con --datos ohiot1dm")
    p.add_argument("--sujeto", default="", help="qué sujeto de OhioT1DM; por omisión el primero")
    p.add_argument("--dias", type=int, default=2, help="cuántos días exportar; la fase 5 pide 2")
    p.add_argument("--semilla", type=int, default=5)
    p.add_argument("--salida", default="serie-para-healthkit.csv")
    p.add_argument("--sin-huecos", action="store_true", help="serie perfecta; para depurar")
    p.add_argument("--sin-retrasos", action="store_true")
    args = p.parse_args()

    puntos = list(
        con_huecos_y_retrasos(
            _lecturas_del_origen(args),
            semilla=args.semilla,
            con_huecos=not args.sin_huecos,
            con_retrasos=not args.sin_retrasos,
        )
    )
    escribir(args.salida, puntos)
    print(f"{args.salida}: {resumen(puntos)}")
    if args.datos != "ohiot1dm":
        print(
            "AVISO: la serie es sintética. Sirve para validar la plomería de HealthKit, no "
            "el comportamiento de un sensor comercial. El reporte tiene que decirlo (R-8)."
        )


if __name__ == "__main__":
    main()
