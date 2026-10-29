"""Pruebas del lector de OhioT1DM.

**Qué comprueban y qué no.** El esquema exacto de los XML no está publicado y el conjunto
todavía no llega, así que estas pruebas construyen archivos con el esquema que `SECCIONES`
reconstruye. Comprueban que el lector haga lo correcto **con ese esquema**, no que el esquema
sea el del conjunto. Lo segundo se comprueba corriendo `esquema_encontrado()` contra el primer
archivo real, y es lo primero que hay que hacer cuando lleguen.

Lo que sí valen desde hoy: las decisiones del lector —qué revienta, qué se ignora, qué se
ordena— no cambian cuando se corrija la tabla, y son las que arruinarían los datos en
silencio.
"""

from datetime import datetime

import pytest

from src.ingest.ohiot1dm import (
    SECCIONES,
    esquema_encontrado,
    leer_archivo,
    leer_carpeta,
    leer_fecha,
)

XML = """<patient id="559" weight="99" insulin_type="Novolog">
  <glucose_level>
    <event ts="07-12-2021 00:05:00" value="138"/>
    <event ts="07-12-2021 00:00:00" value="142"/>
    <event ts="07-12-2021 00:10:00" value="131"/>
  </glucose_level>
  <finger_stick>
    <event ts="07-12-2021 08:30:00" value="95"/>
  </finger_stick>
  <bolus>
    <event ts_begin="07-12-2021 08:35:00" ts_end="07-12-2021 08:35:00" type="normal" dose="4.5"/>
  </bolus>
  <meal>
    <event ts="07-12-2021 08:40:00" type="breakfast" carbs="45"/>
  </meal>
  <exercise>
    <event ts="07-12-2021 18:00:00" intensity="6" duration="30"/>
  </exercise>
  <sleep>
    <event ts_begin="07-12-2021 23:00:00" ts_end="08-12-2021 07:00:00" quality="3"/>
  </sleep>
  <hypo_event>
    <event ts="07-12-2021 03:20:00"/>
  </hypo_event>
  <illness>
  </illness>
</patient>
"""


@pytest.fixture
def archivo(tmp_path):
    ruta = tmp_path / "559-ws-training.xml"
    ruta.write_text(XML, encoding="utf-8")
    return ruta


# --- Lo básico -------------------------------------------------------------------------------


def test_las_lecturas_salen_ordenadas_en_el_tiempo(archivo):
    """Vienen desordenadas en el XML a propósito.

    El resto del repositorio da por hecho que están ordenadas —los rezagos y las derivadas no
    tienen sentido de otra forma— y si no lo estuvieran saldrían números plausibles y
    equivocados en lugar de un error.
    """
    registro = leer_archivo(archivo)
    assert [f.mg_dl for f in registro.lecturas] == [142.0, 138.0, 131.0, 95.0]
    assert [f.ts for f in registro.lecturas] == sorted(f.ts for f in registro.lecturas)


def test_el_sujeto_sale_del_atributo_de_la_raiz(archivo):
    registro = leer_archivo(archivo)
    assert registro.sujeto == "559"
    assert all(f.sujeto == "559" for f in registro.lecturas)


def test_sin_atributo_en_la_raiz_el_sujeto_sale_del_nombre_del_archivo(tmp_path):
    """El respaldo importa: un sujeto mal identificado rompe la partición por persona."""
    ruta = tmp_path / "563-ws-testing.xml"
    ruta.write_text(XML.replace(' id="559"', ""), encoding="utf-8")
    assert leer_archivo(ruta).sujeto == "563"


def test_el_sensor_y_el_pinchazo_se_distinguen(archivo):
    """El juego denso solo se puede construir sobre el sensor.

    Tres pinchazos sueltos no son una serie, y sus rezagos y derivadas serían basura con
    apariencia de característica.
    """
    registro = leer_archivo(archivo)
    del_sensor = [f.mg_dl for f in registro.lecturas if f.del_sensor]
    de_pinchazo = [f.mg_dl for f in registro.lecturas if not f.del_sensor]
    assert del_sensor == [142.0, 138.0, 131.0]
    assert de_pinchazo == [95.0]


def test_los_eventos_quedan_en_su_seccion_con_su_valor(archivo):
    registro = leer_archivo(archivo)
    assert [e.valor for e in registro.eventos["meal"]] == [45.0]
    assert [e.valor for e in registro.eventos["bolus"]] == [4.5]
    assert [e.valor for e in registro.eventos["exercise"]] == [6.0]


def test_las_escalas_de_ohiot1dm_se_dejan_como_vienen(archivo):
    """Ejercicio 1-10 y sueño 1-3, sin traducir.

    Son las del conjunto y por eso los campos de la app se diseñaron así; convertirlas a
    porcentajes o a etiquetas introduce error en los dos lados.
    """
    registro = leer_archivo(archivo)
    assert registro.eventos["exercise"][0].valor == 6.0
    assert registro.eventos["sleep"][0].valor == 3.0


def test_un_evento_sin_valor_numerico_conserva_su_marca_de_tiempo(archivo):
    """Un episodio de hipoglucemia declarado: el dato es que ocurrió."""
    registro = leer_archivo(archivo)
    hipo = registro.eventos["hypo_event"]
    assert len(hipo) == 1
    assert hipo[0].valor is None
    assert hipo[0].ts == datetime(2021, 12, 7, 3, 20)


def test_los_eventos_con_periodo_guardan_su_fin(archivo):
    registro = leer_archivo(archivo)
    sueno = registro.eventos["sleep"][0]
    assert sueno.ts == datetime(2021, 12, 7, 23, 0)
    assert sueno.fin == datetime(2021, 12, 8, 7, 0)


def test_los_atributos_crudos_se_conservan(archivo):
    """`type="breakfast"` no es una característica hoy, y descartarlo sería irreversible."""
    registro = leer_archivo(archivo)
    assert registro.eventos["meal"][0].atributos["type"] == "breakfast"


# --- Lo que podría arruinar los datos en silencio --------------------------------------------


def test_una_fecha_ilegible_revienta(tmp_path):
    """Una fecha que falla en silencio manda la fila a 1970 y la deja fuera de toda ventana."""
    ruta = tmp_path / "559-ws-training.xml"
    ruta.write_text(
        '<patient id="559"><glucose_level>'
        '<event ts="2021/12/07 00:05" value="138"/>'
        "</glucose_level></patient>",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no pude leer la fecha"):
        leer_archivo(ruta)


def test_la_fecha_se_lee_como_dia_mes_ano():
    """`07-12-2021` es el 7 de diciembre, no el 12 de julio.

    Los dos formatos aceptan la misma cadena cuando el día es 12 o menos, así que elegir mal
    desplaza medio conjunto de fecha sin que nada falle. Esta prueba fija cuál se eligió.
    """
    assert leer_fecha("07-12-2021 00:05:00") == datetime(2021, 12, 7, 0, 5)
    assert leer_fecha("07-12-2021 00:05") == datetime(2021, 12, 7, 0, 5)


def test_una_lectura_sin_valor_se_descarta_y_no_se_vuelve_cero(tmp_path):
    """Una glucosa vacía no es una glucosa de cero.

    El artículo avisa de que algunos campos vienen vacíos. Un cero pasaría la lectura, y
    después la validación de 20-600 lo mandaría a cuarentena o, peor, entraría al
    entrenamiento arrastrando el promedio.
    """
    ruta = tmp_path / "559-ws-training.xml"
    ruta.write_text(
        '<patient id="559"><glucose_level>'
        '<event ts="07-12-2021 00:00:00" value="142"/>'
        '<event ts="07-12-2021 00:05:00" value=""/>'
        '<event ts="07-12-2021 00:10:00" value="131"/>'
        "</glucose_level></patient>",
        encoding="utf-8",
    )
    registro = leer_archivo(ruta)
    assert [f.mg_dl for f in registro.lecturas] == [142.0, 131.0]


def test_una_seccion_desconocida_se_ignora_sin_tumbar_la_lectura(tmp_path):
    """Un conjunto que gana un campo en una versión nueva no tiene por qué romper nada."""
    ruta = tmp_path / "559-ws-training.xml"
    nueva = '<cosa_nueva><event ts="1" x="2"/></cosa_nueva>\n  <illness>'
    ruta.write_text(XML.replace("<illness>", nueva), encoding="utf-8")
    registro = leer_archivo(ruta)
    assert len(registro.lecturas) == 4
    assert "cosa_nueva" not in registro.eventos


def test_las_secciones_vacias_se_pueden_consultar(archivo):
    """Varias señales de la banda existen solo en algunos sujetos.

    Un promedio calculado sobre los que sí la traen no se puede presentar como si fuera de los
    doce, y el reporte tiene que poder decir cuáles faltaban.
    """
    registro = leer_archivo(archivo)
    assert "illness" in registro.secciones_vacias
    assert "meal" not in registro.secciones_vacias
    assert "basis_heart_rate" in registro.secciones_vacias


def test_todas_las_secciones_declaradas_aparecen_en_el_registro(archivo):
    """Aunque el archivo no las traiga, para que nadie tenga que usar `.get(...)`."""
    registro = leer_archivo(archivo)
    assert sorted(registro.eventos) == sorted(SECCIONES)


# --- El reconocedor de esquema, que es la herramienta para cuando lleguen los datos ----------


def test_esquema_encontrado_informa_y_no_interpreta(archivo):
    encontrado = esquema_encontrado(archivo)
    assert encontrado["raiz"] == "patient"
    assert encontrado["atributos_de_la_raiz"]["id"] == "559"
    assert encontrado["secciones"]["glucose_level"]["eventos"] == 3
    assert encontrado["secciones"]["glucose_level"]["atributos_del_primero"] == ["ts", "value"]
    assert encontrado["secciones_no_esperadas"] == []
    assert "basis_steps" in encontrado["secciones_esperadas_que_faltan"]


def test_esquema_encontrado_delata_una_seccion_que_la_tabla_no_conoce(tmp_path):
    """Es exactamente para lo que existe: decir en qué no cuadra la reconstrucción."""
    ruta = tmp_path / "559-ws-training.xml"
    ruta.write_text(
        '<patient id="559"><glucose_value><event ts="07-12-2021 00:00:00" v="142"/>'
        "</glucose_value></patient>",
        encoding="utf-8",
    )
    encontrado = esquema_encontrado(ruta)
    assert encontrado["secciones_no_esperadas"] == ["glucose_value"]
    assert encontrado["secciones"]["glucose_value"]["ejemplo"]["v"] == "142"


# --- La carpeta -----------------------------------------------------------------------------


def test_la_carpeta_se_lee_en_orden_de_nombre(tmp_path):
    """Dos corridas que leen en distinto orden pueden partir distinto."""
    for sujeto in ("563", "559", "570"):
        (tmp_path / f"{sujeto}-ws-training.xml").write_text(
            XML.replace('id="559"', f'id="{sujeto}"'), encoding="utf-8"
        )
    assert [r.sujeto for r in leer_carpeta(tmp_path)] == ["559", "563", "570"]


def test_una_carpeta_sin_xml_no_da_registros(tmp_path):
    assert list(leer_carpeta(tmp_path)) == []
