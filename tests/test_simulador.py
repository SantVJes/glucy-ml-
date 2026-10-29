"""Pruebas del lector del simulador fisiológico.

El simulador es el **plan B acordado** si OhioT1DM no llega a tiempo, y a diferencia del
generador de juguete sus métricas sí se pueden reportar diciendo de dónde salen. Eso sube la
exigencia: aquí lo que se comprueba no es que las series existan, sino las tres cosas que
harían que un número del simulador llegara al documento final significando otra cosa.

1. Que la **lista de sujetos siga la regla de edad del perfil de Glucy** y no una selección
   escrita a mano que nadie vuelva a mirar.
2. Que la semilla de cada sujeto dependa de **quién es**, no del orden en que se pidió, porque
   si no, repartir la tanda en varios procesos cambiaría el conjunto.
3. Que el viaje a disco y de vuelta **no pierda ni reetiquete nada**, y que un archivo de otra
   procedencia no se pueda leer como si fuera del simulador.

Las dos primeras necesitan simglucose instalado; la tercera no, y por eso corre también en la
integración continua, que solo instala pytest y ruff.
"""

from datetime import datetime, timedelta

import pytest

from src.datos.simulador import (
    ADOLESCENTES_DESDE_15,
    ADULTOS,
    CARPETA_POR_OMISION,
    INTERVALO_MIN,
    ORIGEN,
    SENSOR,
    SUJETOS,
    cargar,
    guardar,
)
from src.ingest.ohiot1dm import Evento, Lectura, Registro

EDAD_MINIMA = 15  # la del perfil de Glucy; ver las constantes del dominio en glucy-ios


# ---------------------------------------------------------------------------
# La lista de sujetos
# ---------------------------------------------------------------------------


def test_los_sujetos_son_exactamente_los_que_cumplen_la_edad_minima():
    """La lista no se escribió a mano: es la regla de edad del perfil, aplicada.

    Se comprueba en los dos sentidos. Que no sobre nadie evita entrenar con niños de siete
    años, que no son usuarios de la app. Que no falte nadie evita tirar sujetos sin querer, y
    con doce personas en el conjunto real cada sujeto perdido es ocho por ciento del conjunto.
    """
    pd = pytest.importorskip("pandas")
    pkg_resources = pytest.importorskip("pkg_resources")
    archivo = pkg_resources.resource_filename("simglucose", "params/Quest.csv")
    edades = dict(pd.read_csv(archivo)[["Name", "Age"]].itertuples(index=False, name=None))

    assert set(SUJETOS) <= set(edades), "hay sujetos en la lista que simglucose no tiene"
    for sujeto in SUJETOS:
        assert edades[sujeto] >= EDAD_MINIMA, f"{sujeto} tiene {edades[sujeto]} años"
    deberian = {s for s, edad in edades.items() if edad >= EDAD_MINIMA}
    assert set(SUJETOS) == deberian, f"faltan o sobran: {deberian ^ set(SUJETOS)}"


def test_no_hay_sujetos_repetidos():
    """Un sujeto dos veces en la lista aparecería en el entrenamiento y en la prueba del mismo
    pliegue, y la métrica dejaría de medir generalización sin que nada fallara."""
    assert len(SUJETOS) == len(set(SUJETOS))
    assert SUJETOS == ADULTOS + ADOLESCENTES_DESDE_15


# ---------------------------------------------------------------------------
# Las semillas
# ---------------------------------------------------------------------------


def test_la_semilla_depende_del_sujeto_y_no_del_orden_en_que_se_pidio():
    """Es lo que deja simular la tanda en varios procesos sin que el conjunto cambie.

    Si la semilla dependiera del orden de la llamada, `--solo adult#003` daría una serie
    distinta a la que da la tanda completa, y el conjunto en disco dependería de con cuántos
    procesos se generó.
    """
    pytest.importorskip("simglucose")
    from src.datos.simulador import conjunto, uno

    tanda = list(conjunto(3, dias=1))
    suelto = uno(SUJETOS[2], dias=1)
    assert suelto == tanda[2]


def test_un_sujeto_que_no_esta_en_la_lista_revienta():
    """Antes se simulaba, no después: un nombre mal escrito que cae en el `else` del
    controlador de simglucose entrena con los parámetros de un paciente promedio y el archivo
    sale con el nombre pedido, así que nadie lo nota."""
    pytest.importorskip("simglucose")
    from src.datos.simulador import uno

    with pytest.raises(ValueError, match="child#001"):
        uno("child#001", dias=1)


# ---------------------------------------------------------------------------
# El viaje a disco, que es lo que leen los tres entrenamientos
# ---------------------------------------------------------------------------


def _registro_de_prueba(sujeto: str = "adult#001", cuantas: int = 6) -> Registro:
    """Un registro a mano, para no tener que simular en una prueba de serialización."""
    inicio = datetime(2026, 1, 1, 8, 0, 0)
    return Registro(
        sujeto=sujeto,
        lecturas=tuple(
            Lectura(
                sujeto=sujeto,
                ts=inicio + timedelta(minutes=INTERVALO_MIN * i),
                mg_dl=110.0 + i,
                del_sensor=True,
            )
            for i in range(cuantas)
        ),
        eventos={
            "meal": (Evento(sujeto=sujeto, seccion="meal", ts=inicio, valor=45.0),),
            "bolus": (Evento(sujeto=sujeto, seccion="bolus", ts=inicio, valor=4.5),),
            "basal": (Evento(sujeto=sujeto, seccion="basal", ts=inicio, valor=0.013),),
        },
    )


def test_ida_y_vuelta_por_disco_no_pierde_nada(tmp_path):
    """Los tres entrenamientos leen de disco, no simulan. Si el viaje perdiera algo, M-1, M-2 y
    M-3 estarían entrenados con algo que nadie midió."""
    original = _registro_de_prueba()
    guardar([original], tmp_path)
    (vuelta,) = cargar(tmp_path)

    assert vuelta.sujeto == original.sujeto
    assert vuelta.lecturas == original.lecturas
    assert vuelta.eventos["meal"] == original.eventos["meal"]
    assert vuelta.eventos["bolus"] == original.eventos["bolus"]
    assert vuelta.eventos["basal"] == original.eventos["basal"]


def test_las_comidas_y_los_bolos_se_guardan_por_separado(tmp_path):
    """Glucy distingue bolo de basal (RF-14, `HKMetadataKeyInsulinDeliveryReason`), y
    `show_history()` de simglucose los suma en una sola columna. De ahí el bucle propio del
    módulo, y de ahí esta prueba."""
    guardar([_registro_de_prueba()], tmp_path)
    (vuelta,) = cargar(tmp_path)
    assert set(vuelta.eventos) >= {"meal", "bolus", "basal"}
    assert vuelta.eventos["bolus"][0].valor != vuelta.eventos["basal"][0].valor


def test_una_carpeta_vacia_revienta_en_lugar_de_devolver_cero_sujetos(tmp_path):
    """Un entrenamiento sobre cero filas falla mucho más adelante y con un mensaje que no
    apunta a la carpeta."""
    with pytest.raises(FileNotFoundError, match="python -m src.datos.simulador"):
        cargar(tmp_path)


def test_un_archivo_de_otro_origen_no_se_lee_como_del_simulador(tmp_path):
    """Es la guardia que impide que el manifiesto diga «simulador» sobre datos que no lo son.

    El manifiesto existe para que ningún número llegue al documento final sin su etiqueta; un
    JSON ajeno leído aquí saldría etiquetado como del simulador y la etiqueta dejaría de valer.
    """
    import json

    (tmp_path / "ajeno.json").write_text(
        json.dumps({"sujeto": "adult#001", "origen": "sintetico", "lecturas": [], "eventos": {}})
    )
    with pytest.raises(ValueError, match="sintetico"):
        cargar(tmp_path)


def test_el_nombre_del_archivo_no_lleva_almohadilla(tmp_path):
    """`adult#001` con almohadilla en el nombre es un ancla en una URL y un comentario en media
    herramienta de línea de órdenes."""
    (ruta,) = guardar([_registro_de_prueba("adolescent#007")], tmp_path)
    assert "#" not in ruta.name
    assert ruta.name == "adolescent-007.json"
    (vuelta,) = cargar(tmp_path)
    assert vuelta.sujeto == "adolescent#007"


# ---------------------------------------------------------------------------
# Las constantes que el resto del repositorio da por buenas
# ---------------------------------------------------------------------------


def test_el_muestreo_es_de_cinco_minutos():
    """Toda la tubería de Glucy supone cinco minutos: los rezagos a 5/10/15/30/60 min, la
    interpolación de huecos de hasta 15 min y la frescura del sensor. El Dexcom de simglucose
    muestrea cada tres y el Navigator cada uno; por eso el sensor es el GuardianRT."""
    assert INTERVALO_MIN == 5
    assert SENSOR == "GuardianRT"


def test_el_origen_es_el_que_entiende_el_manifiesto():
    """`artefacto.ORIGENES` decide qué se puede reportar. Si esta cadena no coincidiera, el
    manifiesto reventaría al guardar, pero después de la hora de entrenamiento."""
    from src.train.artefacto import ORIGENES

    assert ORIGEN in ORIGENES
    assert ORIGEN == "simulador"
    assert CARPETA_POR_OMISION.startswith("data/")
