"""Lo que el cuaderno de evaluación necesita antes de dibujar, y la orden que lo deja todo.

El cuaderno (`notebooks/evaluacion.ipynb`) es un entregable del reporte, y un cuaderno es el
peor sitio donde dejar lógica: no se prueba, no pasa por el estilo y se rompe sin que nadie se
entere hasta que alguien lo abre. Así que lo que decide algo vive aquí, con sus pruebas, y el
cuaderno solo llama y enseña.

`main()` hace lo mismo sin abrir Jupyter: deja las cuatro figuras en `artifacts/figuras/`. Es lo
que se corre para armar el documento.

## El cuaderno tiene que correr hoy

Hoy no hay datos de OhioT1DM ni modelo entrenado, y faltan semanas para que los haya. Un
cuaderno que no se pueda abrir hasta entonces es un cuaderno que se escribe de prisa al final.
Así que si no hay conjunto, se usa el generador de juguete, y **si se usa, se dice en cada
figura y en la salida**: una corrida sintética no es un resultado, y una figura sin esa
advertencia acaba pegada en un documento como si lo fuera (riesgo R-8).
"""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from src.eval.baselines import persistencia
from src.eval.metrics import Resultado, evaluar, rmse

# El horizonte principal del proyecto y los pasos de cinco minutos que lo componen.
HORIZONTE_MIN = 30
PASOS_POR_HORIZONTE = HORIZONTE_MIN // 5

AVISO_SINTETICO = (
    "serie sintética: no es un resultado, es una comprobación de que el cuaderno corre (R-8)"
)


@dataclass(frozen=True)
class Evaluacion:
    """Lo que el cuaderno dibuja, ya calculado.

    Lleva `origen` y `es_reportable` porque la diferencia entre una corrida de prueba y un
    resultado del documento final no se puede dejar a la memoria de quien abre el cuaderno.
    """

    reales: list[float]
    predichos: list[float]
    minutos: list[float]
    origen: str
    resultado: Resultado
    rmse_por_modelo: dict[str, float]

    @property
    def es_reportable(self) -> bool:
        return self.origen == "ohiot1dm"

    @property
    def subtitulo(self) -> str:
        return f"{self.origen}" if self.es_reportable else AVISO_SINTETICO


def serie_de_evaluacion(carpeta: Path | str = "data", *, dias: int = 5) -> tuple[list[float], str]:
    """La serie del sensor con que se evalúa, y de dónde salió.

    Prefiere OhioT1DM y cae al generador de juguete. **La caída se devuelve, no se imprime y
    ya**: quien llama tiene que poder marcar la figura, y un `print` se pierde entre la salida
    de una celda.
    """
    ruta = Path(carpeta)
    if ruta.is_dir() and any(ruta.glob("*.xml")):
        from src.ingest.ohiot1dm import leer_carpeta

        registros = list(leer_carpeta(ruta))
        if registros:
            lecturas = sorted(
                (lec for lec in registros[0].lecturas if lec.del_sensor), key=lambda f: f.ts
            )
            if lecturas:
                return [lec.mg_dl for lec in lecturas], "ohiot1dm"

    from src.datos.sintetico import conjunto

    registro = next(iter(conjunto(1, dias=dias, semilla=4)))
    lecturas = sorted((lec for lec in registro.lecturas if lec.del_sensor), key=lambda f: f.ts)
    return [lec.mg_dl for lec in lecturas], "sintetico"


def evaluacion_de_referencia(serie: Sequence[float], origen: str) -> Evaluacion:
    """La evaluación de la **persistencia**, que es lo que se puede calcular sin modelo.

    No es un relleno. La persistencia es la línea base que el protocolo pone como piso —«un
    modelo que no le gana no justifica su existencia»—, así que esta es exactamente la figura
    contra la que se va a comparar M-1 el día que exista. Tenerla dibujada de antemano es tener
    el listón puesto antes de medir, que es el orden correcto.
    """
    if len(serie) <= PASOS_POR_HORIZONTE:
        raise ValueError(
            f"la serie tiene {len(serie)} lecturas: no alcanza para {HORIZONTE_MIN} min"
        )

    reales = list(serie[PASOS_POR_HORIZONTE:])
    predichos = persistencia(serie, PASOS_POR_HORIZONTE)[: len(reales)]
    return Evaluacion(
        reales=reales,
        predichos=predichos,
        minutos=[i * 5 for i in range(len(reales))],
        origen=origen,
        resultado=evaluar(reales, predichos, horizonte_min=HORIZONTE_MIN),
        rmse_por_modelo={"persistencia": rmse(reales, predichos)},
    )


def escribir_figuras(evaluacion: Evaluacion, carpeta: Path | str) -> dict[str, Path]:
    """Las cuatro figuras en disco, con el origen escrito en cada título."""
    from src.eval import dibujos

    carpeta = Path(carpeta)
    sufijo = f" · {evaluacion.subtitulo}"
    hasta = min(len(evaluacion.minutos), 72)  # seis horas: más no se lee en una figura

    escritas: dict[str, Path] = {}
    figura = dibujos.rejilla_de_clarke(
        evaluacion.reales,
        evaluacion.predichos,
        titulo=f"Rejilla de Clarke a {HORIZONTE_MIN} min{sufijo}",
    )
    escritas["clarke"] = dibujos.guardar(figura, carpeta / "clarke.png")

    figura = dibujos.medido_contra_predicho(
        evaluacion.minutos[:hasta],
        evaluacion.reales[:hasta],
        evaluacion.predichos[:hasta],
        titulo=f"Lo medido y lo predicho a {HORIZONTE_MIN} min{sufijo}",
    )
    escritas["serie"] = dibujos.guardar(figura, carpeta / "serie.png")

    figura = dibujos.contra_las_lineas_base(
        evaluacion.rmse_por_modelo,
        el_del_proyecto=min(evaluacion.rmse_por_modelo, key=evaluacion.rmse_por_modelo.get),
    )
    escritas["lineas-base"] = dibujos.guardar(figura, carpeta / "lineas-base.png")

    # El desglose por origen necesita predicciones sobre lecturas manuales y por foto, y eso
    # sale del pipeline con datos de verdad. Mientras no los haya, la figura se arma con el
    # único origen que hay, para que el cuaderno corra entero y se vea la forma que tendrá.
    figura = dibujos.error_por_origen(
        {"sensor": evaluacion.resultado.mae},
        n_por_origen={"sensor": evaluacion.resultado.n},
    )
    escritas["origen"] = dibujos.guardar(figura, carpeta / "error-por-origen.png")
    return escritas


def main() -> None:
    p = argparse.ArgumentParser(
        description="Deja las cuatro figuras del cuaderno de evaluación en una carpeta."
    )
    p.add_argument("--carpeta", default="data", help="dónde están los XML de OhioT1DM")
    p.add_argument("--salida", default="artifacts/figuras")
    args = p.parse_args()

    serie, origen = serie_de_evaluacion(args.carpeta)
    evaluacion = evaluacion_de_referencia(serie, origen)
    escritas = escribir_figuras(evaluacion, args.salida)

    for nombre, ruta in escritas.items():
        print(f"{nombre}: {ruta}")
    print(
        f"persistencia a {HORIZONTE_MIN} min · A+B {evaluacion.resultado.clarke_ab:.1f} % · "
        f"RMSE {evaluacion.resultado.rmse:.1f} mg/dL · n = {evaluacion.resultado.n}"
    )
    if not evaluacion.es_reportable:
        print(f"AVISO: {AVISO_SINTETICO}")


if __name__ == "__main__":
    main()
