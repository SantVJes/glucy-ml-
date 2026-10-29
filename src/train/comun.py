"""Lo que comparten los tres guiones de entrenamiento: de dónde salen los datos y cómo se corre.

Está aparte para que `train_m1.py`, `train_m2.py` y `train_m3.py` sean cada uno veinte líneas
que dicen **qué** entrenan, y no tres copias de **cómo** se entrena. Tres copias de un bucle de
evaluación es como dos de ellas acaban midiendo distinto que la tercera sin que nadie lo note.
"""

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from src.datos import sintetico
from src.features.ventanas import Fila, filas_densas
from src.ingest.ohiot1dm import Registro, leer_carpeta
from src.train.artefacto import guardar, manifiesto_de
from src.train.entrenar import Corrida, evaluar_con_loso, resumen
from src.train.modelos import LSTM, Bosque, ExtrapolacionLineal, Persistencia, Ridge, XGBoost

SEMILLA_POR_OMISION = int(os.environ.get("GLUCY_SEMILLA", "42"))
CARPETA_DATOS = os.environ.get("GLUCY_RUTA_DATOS", "data/ohiot1dm")
CARPETA_ARTEFACTOS = os.environ.get("GLUCY_RUTA_ARTEFACTOS", "artifacts")


def argumentos(descripcion: str) -> argparse.ArgumentParser:
    """Los argumentos que comparten los tres guiones.

    `--datos sintetico` existe para poder correr el guion completo **sin** el conjunto de datos,
    y por eso el manifiesto que sale marca el origen: una métrica de una corrida sintética no se
    puede reportar y el archivo lo dice.
    """
    p = argparse.ArgumentParser(description=descripcion)
    p.add_argument(
        "--datos",
        default="ohiot1dm",
        choices=("ohiot1dm", "sintetico"),
        help="de dónde salen las series. 'sintetico' es para probar el guion, no para reportar",
    )
    p.add_argument("--carpeta", default=CARPETA_DATOS, help="carpeta de los XML de OhioT1DM")
    p.add_argument("--version", default="v0.1.0", help="versión del artefacto que se guarda")
    p.add_argument("--semilla", type=int, default=SEMILLA_POR_OMISION)
    p.add_argument(
        "--sujetos",
        type=int,
        default=12,
        help="cuántas personas sintéticas generar; se ignora con --datos ohiot1dm",
    )
    p.add_argument(
        "--dias", type=int, default=14, help="días por persona sintética; se ignora con ohiot1dm"
    )
    p.add_argument(
        "--umbral-hipo",
        type=int,
        default=70,
        help="umbral de hipoglucemia para la sensibilidad y las falsas alarmas",
    )
    p.add_argument(
        "--sin-guardar",
        action="store_true",
        help="evalúa e imprime, pero no escribe el artefacto",
    )
    p.add_argument(
        "--sin-lstm",
        action="store_true",
        help=(
            "salta la línea base recurrente, que es la lenta. Para iterar, no para reportar: "
            "el protocolo pide las cinco"
        ),
    )
    return p


def registros(args: argparse.Namespace) -> list[Registro]:
    """Las series, de los XML o del generador de juguete.

    Con `ohiot1dm` revienta si la carpeta está vacía en lugar de caer al generador: una corrida
    que dice «ohiot1dm» en el manifiesto y entrenó con senos sería el peor error posible de este
    repositorio, porque el número llegaría al documento final sin nada que lo delate.
    """
    if args.datos == "sintetico":
        return list(sintetico.conjunto(args.sujetos, dias=args.dias, semilla=args.semilla))

    carpeta = Path(args.carpeta)
    if not carpeta.is_dir():
        raise SystemExit(
            f"no existe la carpeta {carpeta}. El conjunto no está en el repositorio: hay que "
            "pedirlo y descomprimirlo ahí (ver data/README.md). Para probar el guion sin datos, "
            "--datos sintetico."
        )
    leidos = list(leer_carpeta(carpeta))
    if not leidos:
        raise SystemExit(f"la carpeta {carpeta} no tiene ningún .xml que leer")
    return leidos


def filas_de(
    registros_: Sequence[Registro], *, horizonte_min: int, absorcion_min: float = 180
) -> list[Fila]:
    """Las filas del juego denso de todos los sujetos, con su objetivo al horizonte pedido."""
    filas: list[Fila] = []
    for r in registros_:
        filas.extend(
            filas_densas(
                r.lecturas,
                r.eventos.get("meal", ()),
                r.eventos.get("bolus", ()),
                horizonte_min=horizonte_min,
                absorcion_min=absorcion_min,
            )
        )
    if not filas:
        raise SystemExit(
            f"ninguna fila quedó con objetivo a {horizonte_min} min. Si las series son muy "
            "cortas o tienen huecos muy grandes, no hay nada que entrenar."
        )
    return filas


def _lineas_base(args: argparse.Namespace, horizonte_min: int) -> dict[str, object]:
    """Las cinco del protocolo. La recurrente va **encendida por omisión**, a propósito.

    El protocolo las pide todas, y una tabla a la que le falta una línea base es una tabla que
    hay que explicar en el reporte. Que la rápida sea la incompleta y no al revés: `--sin-lstm`
    está para iterar, y quien lo escribe sabe lo que se está saltando.
    """
    lineas: dict[str, object] = {
        "persistencia": Persistencia(),
        "extrapolacion_lineal": ExtrapolacionLineal(horizonte_min),
        "ridge": Ridge(semilla=args.semilla),
        "bosque": Bosque(semilla=args.semilla),
    }
    if not getattr(args, "sin_lstm", False):
        lineas["lstm"] = LSTM(semilla=args.semilla)
    return lineas


def correr(
    args: argparse.Namespace,
    *,
    modelo: str,
    horizonte_min: int,
    juego: str = "denso",
) -> Corrida:
    """Entrena, evalúa contra las líneas base, imprime el cuadro y guarda el artefacto.

    Las líneas base se construyen aquí y no en cada guion para que **los tres modelos se
    comparen contra lo mismo**. La extrapolación lineal toma su ventana de pendiente igual al
    horizonte, que es la versión defendible de esa línea base.
    """
    from src.features.features import manifiesto_de_caracteristicas

    filas = filas_de(registros(args), horizonte_min=horizonte_min)
    print(f"{len(filas)} filas de {len({f.sujeto for f in filas})} sujetos")

    corrida = evaluar_con_loso(
        filas,
        juego=juego,
        horizonte_min=horizonte_min,
        construir_modelo=lambda: XGBoost(semilla=args.semilla),
        construir_lineas_base=lambda: _lineas_base(args, horizonte_min),
        umbral_hipo=args.umbral_hipo,
        nombre=modelo,
    )
    print()
    print(resumen(corrida))
    print()
    if not corrida.le_gana_a_la_persistencia:
        print(
            "AVISO: este modelo no le gana a la persistencia, así que no justifica su "
            "existencia. Antes de tocar hiperparámetros, revisar las características."
        )
    if args.datos != "ohiot1dm":
        print(
            f"AVISO: los datos son '{args.datos}'. Estas métricas NO se pueden reportar: "
            "la señal es sintética y por construcción predecible."
        )

    if args.sin_guardar:
        return corrida

    # El artefacto final se entrena con **todos** los sujetos: los pliegues sirven para medir,
    # no para elegir el modelo que se publica. Medir con LOSO y publicar el de un pliegue sería
    # publicar un modelo que nunca vio a una de las doce personas, por nada.
    x_todo = _matriz(filas, juego)
    y_todo = [f.objetivo for f in filas]
    definitivo = XGBoost(semilla=args.semilla)
    definitivo.ajustar(x_todo, y_todo)

    manifiesto = manifiesto_de(
        modelo=modelo,
        version=args.version,
        algoritmo=definitivo.algoritmo,
        horizonte_min=horizonte_min,
        juego=juego,
        caracteristicas=manifiesto_de_caracteristicas(juego),
        semilla=args.semilla,
        hiperparametros=definitivo.hiperparametros,
        origen_datos=args.datos,
        n_entrenamiento=len(filas),
        resultado=corrida.resultado,
        sujetos_entrenamiento=list(corrida.sujetos),
        sujetos_prueba=[],
        protocolo=(
            f"Validación cruzada dejando un sujeto fuera sobre {len(corrida.sujetos)} sujetos; "
            "las métricas son de esos pliegues. El artefacto que se publica se entrenó después "
            "con todos los sujetos, con los mismos hiperparámetros y la misma semilla."
        ),
        lineas_base=corrida.lineas_base,
        por_sujeto=corrida.por_sujeto,
    )
    rutas = guardar(manifiesto, definitivo.guardar, CARPETA_ARTEFACTOS)
    for que, donde in rutas.items():
        print(f"{que}: {donde}")
    return corrida


def _matriz(filas: Sequence[Fila], juego: str):
    from src.features.features import a_matriz

    return a_matriz([f.caracteristicas for f in filas], juego)
