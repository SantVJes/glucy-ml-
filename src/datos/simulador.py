"""Series de un **simulador fisiológico**, que es el respaldo del proyecto si OhioT1DM no llega.

A diferencia de `sintetico.py`, esto no inventa una curva: resuelve un modelo del metabolismo.
Se le dan comidas, un controlador pone la insulina, y la glucosa sale de las ecuaciones. Por eso
un modelo entrenado aquí aprende la relación entre carbohidratos, insulina y glucosa en lugar de
aprenderse un generador, y por eso **sus métricas sí se pueden reportar diciendo que son del
simulador** (`ORIGENES` en `src/train/artefacto.py` lo tiene como escalón propio, distinto de
`sintetico`).

Sigue sin ser el plan: los sujetos son simulados y la evidencia es más débil que la de doce
personas reales. El reporte lo tiene que decir con esas palabras.

## Qué se usa, y por qué

**simglucose** (licencia MIT), que implementa el simulador UVA/Padova de 2008 en Python. Cumple
las tres condiciones del proyecto: es gratis y de código abierto, es Python como el resto de
este repositorio, y trae pacientes virtuales ya parametrizados.

**El sensor es el GuardianRT, que muestrea cada 5 minutos**, igual que OhioT1DM. El Dexcom de
simglucose muestrea cada 3, y todo el pipeline —los rezagos a 5/10/15/30/60, la interpolación de
huecos, la frescura— está escrito sobre una cadencia de cinco. Cambiarla aquí daría un modelo
que no se parece al que correrá con datos reales.

## Por qué 18 sujetos y no los 30 que hay

Los diez niños tienen entre **7 y 12 años** y dos adolescentes tienen **14**. El perfil de Glucy
exige **15 años cumplidos**, así que entrenar con ellos sería medir el sistema sobre gente que
la app declara fuera de su alcance, e inflar el resultado con personas que nunca lo usarían.
Quedan los diez adultos (26 a 68 años) y los ocho adolescentes de 15 o más: **18 sujetos**, la
mitad más que los doce de OhioT1DM, y todos dentro del rango que la app declara.

## Lo que este simulador no reproduce, y va al reporte

El controlador calcula cada bolo **a partir de los carbohidratos exactos** de la comida. Nadie
cuenta así: la persona estima, se equivoca, y de ahí salen buena parte de las hipoglucemias
reales. Aquí no se modela ese error —sería inventarse un número que el simulador no sostiene—,
así que **la variabilidad sale menor que la de una persona de verdad**. Aun así la serie baja:
con un adulto y diez días, 4.5 % del tiempo por debajo de 70 mg/dL y un mínimo de 51, que está
en el orden de lo que se ve en clínica.
"""

import argparse
import json
import warnings
from collections.abc import Iterable, Iterator
from datetime import datetime, timedelta
from pathlib import Path

from src.ingest.ohiot1dm import Evento, Lectura, Registro

ORIGEN = "simulador"

# Cinco minutos, la cadencia de OhioT1DM y la que supone todo el pipeline.
SENSOR = "GuardianRT"
BOMBA = "Insulet"
INTERVALO_MIN = 5

# Los diez adultos y los adolescentes de 15 o más. Ver el encabezado: los niños (7-12) y los
# dos adolescentes de 14 quedan fuera porque el perfil de la app pide 15 años cumplidos.
ADULTOS = tuple(f"adult#{i:03d}" for i in range(1, 11))
ADOLESCENTES_DESDE_15 = (
    "adolescent#001",
    "adolescent#002",
    "adolescent#003",
    "adolescent#004",
    "adolescent#005",
    "adolescent#007",
    "adolescent#009",
    "adolescent#010",
)
SUJETOS = ADULTOS + ADOLESCENTES_DESDE_15

INICIO_POR_OMISION = datetime(2026, 1, 1, 0, 0, 0)


def _piezas(sujeto: str, inicio: datetime, semilla: int):
    """Arma el entorno de simglucose. El import va aquí dentro a propósito.

    La integración continua instala solo pytest y ruff; un import en la cabecera dejaría el
    módulo sin poder cargarse allá y arrastraría al resto de `src.datos`.
    """
    with warnings.catch_warnings():
        # `gym` 0.9.4, que simglucose arrastra, usa `pkg_resources`, que está deprecado.
        warnings.simplefilter("ignore")
        from simglucose.actuator.pump import InsulinPump
        from simglucose.controller.basal_bolus_ctrller import BBController
        from simglucose.patient.t1dpatient import T1DPatient
        from simglucose.sensor.cgm import CGMSensor
        from simglucose.simulation.env import T1DSimEnv
        from simglucose.simulation.scenario_gen import RandomScenario

    entorno = T1DSimEnv(
        T1DPatient.withName(sujeto),
        CGMSensor.withName(SENSOR, seed=semilla),
        InsulinPump.withName(BOMBA),
        RandomScenario(start_time=inicio, seed=semilla),
    )
    return entorno, BBController()


def registro(
    sujeto: str,
    *,
    dias: int = 28,
    semilla: int = 42,
    inicio: datetime | None = None,
) -> Registro:
    """Una persona simulada, con sus lecturas y sus eventos, lista para el pipeline.

    El bucle es propio en lugar de `SimObj` porque el historial de simglucose guarda la insulina
    **sumada**, basal más bolo en una sola columna. Glucy los distingue (RF-14, y el metadato
    `HKMetadataKeyInsulinDeliveryReason` en el teléfono), así que hay que capturar la acción del
    controlador antes de que el entorno las junte.
    """
    if dias <= 0:
        raise ValueError(f"los días tienen que ser al menos uno, no {dias}")
    if sujeto not in SUJETOS:
        raise ValueError(f"sujeto desconocido: {sujeto!r}. Los válidos están en SUJETOS")

    inicio = inicio or INICIO_POR_OMISION
    entorno, controlador = _piezas(sujeto, inicio, semilla)
    fin = inicio + timedelta(days=dias)

    controlador.reset()
    observacion, recompensa, listo, info = entorno.reset()

    lecturas: list[Lectura] = []
    comidas: list[Evento] = []
    bolos: list[Evento] = []
    basales: list[Evento] = []
    basal_anterior: float | None = None

    while entorno.time < fin:
        accion = controlador.policy(observacion, recompensa, listo, **info)
        ts = entorno.time
        observacion, recompensa, listo, info = entorno.step(accion)

        lecturas.append(
            Lectura(sujeto=sujeto, ts=ts, mg_dl=float(observacion.CGM), del_sensor=True)
        )

        # `info["meal"]` viene en g/min sobre el paso de muestreo: los gramos de la comida son
        # el ritmo por los minutos que duró el paso.
        gramos = float(info["meal"]) * INTERVALO_MIN
        if gramos > 0:
            comidas.append(Evento(sujeto=sujeto, seccion="meal", ts=ts, valor=gramos))

        # El bolo también llega como ritmo, por la misma razón: la bomba solo acepta U/min.
        unidades = float(accion.bolus) * INTERVALO_MIN
        if unidades > 0:
            bolos.append(Evento(sujeto=sujeto, seccion="bolus", ts=ts, valor=unidades))

        # La basal es continua; se registra solo cuando cambia, como en OhioT1DM, donde un
        # renglón de `basal` significa «a partir de aquí, esta tasa».
        tasa = float(accion.basal)
        if basal_anterior is None or tasa != basal_anterior:
            basales.append(Evento(sujeto=sujeto, seccion="basal", ts=ts, valor=tasa))
            basal_anterior = tasa

    return Registro(
        sujeto=sujeto,
        lecturas=tuple(lecturas),
        eventos={
            "meal": tuple(comidas),
            "bolus": tuple(bolos),
            "basal": tuple(basales),
        },
    )


def conjunto(
    cuantos: int = len(SUJETOS),
    *,
    dias: int = 28,
    semilla: int = 42,
    inicio: datetime | None = None,
) -> Iterator[Registro]:
    """Los primeros `cuantos` sujetos, cada uno con **su propia semilla**.

    Semillas distintas por persona para que las comidas y el ruido del sensor no coincidan entre
    sujetos. Con la misma semilla para todos, dos personas comerían a la misma hora los mismos
    gramos, y repartir por sujeto dejaría de significar algo: el modelo vería en la prueba el
    mismo horario que memorizó en el entrenamiento.
    """
    if not 1 <= cuantos <= len(SUJETOS):
        raise ValueError(f"entre 1 y {len(SUJETOS)} sujetos, no {cuantos}")
    for sujeto in SUJETOS[:cuantos]:
        yield uno(sujeto, dias=dias, semilla=semilla, inicio=inicio)


def uno(
    sujeto: str,
    *,
    dias: int = 28,
    semilla: int = 42,
    inicio: datetime | None = None,
) -> Registro:
    """Un sujeto con la semilla que le toca por **su posicion en `SUJETOS`**, no por el orden en
    que se pidio.

    Existe para poder simular en varios procesos a la vez sin que el conjunto cambie. Si la
    semilla dependiera del orden de la llamada, `--solo adult#005` daria una serie distinta a la
    que da la tanda completa, y el conjunto en disco dependeria de como se lanzo el guion.
    """
    if sujeto not in SUJETOS:
        raise ValueError(f"{sujeto!r} no esta en SUJETOS. Los validos son {SUJETOS}")
    return registro(sujeto, dias=dias, semilla=semilla + SUJETOS.index(sujeto), inicio=inicio)


# ---------------------------------------------------------------------------
# Guardar y leer, para no volver a simular
# ---------------------------------------------------------------------------
#
# Simular los 18 sujetos cuesta más de una hora. Entrenar M-1, M-2 y M-3 sobre la misma tanda
# lo haría tres veces y daría **tres conjuntos distintos** si alguien tocara una semilla por el
# camino, que es la peor forma de equivocarse: los tres modelos dejarían de ser comparables
# entre sí sin que nada falle. Se simula una vez, se guarda, y los tres leen lo mismo.
#
# No se versiona: `data/` está en `.gitignore` y esto se regenera de la semilla.

CARPETA_POR_OMISION = "data/simulador"


def guardar(registros: Iterable[Registro], carpeta: Path | str) -> list[Path]:
    """Un JSON por sujeto. Devuelve lo que escribió."""
    destino = Path(carpeta)
    destino.mkdir(parents=True, exist_ok=True)
    escritos = []
    for r in registros:
        datos = {
            "sujeto": r.sujeto,
            "origen": ORIGEN,
            "lecturas": [[lec.ts.isoformat(), lec.mg_dl] for lec in r.lecturas],
            "eventos": {
                seccion: [[e.ts.isoformat(), e.valor] for e in eventos]
                for seccion, eventos in r.eventos.items()
            },
        }
        ruta = destino / f"{r.sujeto.replace('#', '-')}.json"
        ruta.write_text(json.dumps(datos))
        escritos.append(ruta)
    return escritos


def cargar(carpeta: Path | str = CARPETA_POR_OMISION) -> list[Registro]:
    """Lee lo que dejó `guardar`. Revienta si la carpeta está vacía.

    Revienta en lugar de devolver una lista vacía porque un entrenamiento sobre cero filas
    falla mucho más adelante y con un mensaje que no apunta aquí.
    """
    origen = Path(carpeta)
    rutas = sorted(origen.glob("*.json"))
    if not rutas:
        raise FileNotFoundError(
            f"no hay series del simulador en {origen}. Generarlas con: "
            f"python -m src.datos.simulador --salida {origen}"
        )

    registros = []
    for ruta in rutas:
        datos = json.loads(ruta.read_text())
        # Se comprueba el origen que trae el archivo en lugar de confiar en la carpeta: un JSON
        # de otra procedencia leido aqui acabaria con `origen_datos: "simulador"` en el
        # manifiesto, que es exactamente la mentira que el manifiesto existe para evitar.
        if datos.get("origen") != ORIGEN:
            raise ValueError(
                f"{ruta} dice que su origen es {datos.get('origen')!r}, no {ORIGEN!r}. "
                "No se lee: el manifiesto del modelo acabaria diciendo que son del simulador."
            )
        sujeto = datos["sujeto"]
        registros.append(
            Registro(
                sujeto=sujeto,
                lecturas=tuple(
                    Lectura(
                        sujeto=sujeto,
                        ts=datetime.fromisoformat(ts),
                        mg_dl=mg_dl,
                        del_sensor=True,
                    )
                    for ts, mg_dl in datos["lecturas"]
                ),
                eventos={
                    seccion: tuple(
                        Evento(
                            sujeto=sujeto,
                            seccion=seccion,
                            ts=datetime.fromisoformat(ts),
                            valor=valor,
                        )
                        for ts, valor in eventos
                    )
                    for seccion, eventos in datos["eventos"].items()
                },
            )
        )
    return registros


def main() -> None:
    p = argparse.ArgumentParser(
        description="Genera las series del simulador fisiológico y las deja en disco."
    )
    p.add_argument("--sujetos", type=int, default=len(SUJETOS))
    p.add_argument("--dias", type=int, default=28)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--salida", default=CARPETA_POR_OMISION)
    p.add_argument(
        "--solo",
        default=None,
        help=(
            "lista separada por comas de sujetos a simular, para repartir la tanda en varios "
            "procesos. El conjunto que sale es el mismo que con la tanda completa"
        ),
    )
    args = p.parse_args()

    if args.solo:
        pedidos = [s.strip() for s in args.solo.split(",") if s.strip()]
        tanda = (uno(s, dias=args.dias, semilla=args.semilla) for s in pedidos)
    else:
        tanda = conjunto(args.sujetos, dias=args.dias, semilla=args.semilla)

    escritos = []
    for r in tanda:
        bajas = sum(1 for lec in r.lecturas if lec.mg_dl < 70)
        print(
            f"{r.sujeto}: {len(r.lecturas)} lecturas · {len(r.eventos['meal'])} comidas · "
            f"{bajas} por debajo de 70 ({100 * bajas / len(r.lecturas):.1f} %)",
            flush=True,
        )
        escritos.extend(guardar([r], args.salida))
    print(f"{len(escritos)} archivos en {args.salida}")


if __name__ == "__main__":
    main()
