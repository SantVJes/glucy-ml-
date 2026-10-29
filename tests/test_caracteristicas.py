"""Pruebas de los dos juegos de características.

La primera es **una prueba de valores literales a propósito**. El backend es la autoridad sobre
estas listas y los dos repositorios no se pueden importar entre sí, así que la única red
posible es que agregar, quitar o mover una característica sea imposible de hacer sin ver fallar
una prueba. Un orden distinto aquí y allá no rompe nada: alimenta el modelo con las columnas
cambiadas de lugar y la predicción sale plausible y equivocada.
"""

import math

import pytest

from src.features.features import (
    CARACTERISTICAS_DENSAS,
    CARACTERISTICAS_DISPERSAS,
    CODIGOS_CONTEXTO,
    FALTA,
    a_matriz,
    a_vector,
    codigo_contexto,
    hora_ciclica,
    manifiesto_de_caracteristicas,
)


def test_el_juego_denso_es_exactamente_este():
    """Copiado de `app/services/pipeline/caracteristicas.py` del backend, en ese orden."""
    assert CARACTERISTICAS_DENSAS == (
        "mg_dl",
        "rezago_5",
        "rezago_10",
        "rezago_15",
        "rezago_30",
        "rezago_60",
        "derivada_1",
        "derivada_2",
        "media_30",
        "media_60",
        "cob",
        "iob",
        "hora_sin",
        "hora_cos",
        "dia_semana",
    )


def test_el_juego_disperso_es_exactamente_este():
    assert CARACTERISTICAS_DISPERSAS == (
        "mg_dl",
        "antiguedad_min",
        "carbs_g",
        "dosis_ui",
        "cob",
        "iob",
        "hora_sin",
        "hora_cos",
        "contexto_comida",
        "promedio_7d",
    )


def test_el_denso_no_trae_actividad_ni_sueno():
    """C-10 y C-11: en producción nunca llegarían, así que entrenar con ellas es peor.

    La prueba es por la tentación obvia: OhioT1DM **sí** trae ejercicio y sueño, y al tener el
    conjunto delante es lo primero que uno quiere agregar. El teléfono no las captura y el
    servidor no tiene endpoint para el contexto; el modelo les daría peso y luego nunca lo
    recibiría.
    """
    assert "actividad" not in CARACTERISTICAS_DENSAS
    assert "sueno" not in CARACTERISTICAS_DENSAS
    assert "sueño" not in CARACTERISTICAS_DENSAS


def test_el_peso_no_entra_en_ningun_juego():
    """El conjunto lo trae como relleno (99 para todos) y es constante a 30 minutos."""
    assert "peso" not in CARACTERISTICAS_DENSAS
    assert "peso" not in CARACTERISTICAS_DISPERSAS
    assert "peso_kg" not in CARACTERISTICAS_DENSAS
    assert "peso_kg" not in CARACTERISTICAS_DISPERSAS


def test_ninguna_caracteristica_se_repite():
    for juego in (CARACTERISTICAS_DENSAS, CARACTERISTICAS_DISPERSAS):
        assert len(juego) == len(set(juego))


# --- La hora cíclica -------------------------------------------------------------------------


def test_la_medianoche_y_las_2350_quedan_cerca():
    """Es el motivo de que la hora sea seno y coseno y no un número de 0 a 23.

    La hipoglucemia nocturna es el evento que importa, y pasa justo en el salto de 23 a 0.
    """
    casi_medianoche = hora_ciclica(23 * 60 + 50)
    pasada = hora_ciclica(10)
    distancia = math.dist(casi_medianoche, pasada)
    media_tarde = math.dist(casi_medianoche, hora_ciclica(12 * 60))
    assert distancia < 0.1
    assert distancia < media_tarde


def test_las_horas_conocidas_dan_los_valores_conocidos():
    sin_0, cos_0 = hora_ciclica(0)
    assert (sin_0, cos_0) == pytest.approx((0.0, 1.0))
    sin_6, cos_6 = hora_ciclica(6 * 60)
    assert (sin_6, cos_6) == pytest.approx((1.0, 0.0))
    sin_12, cos_12 = hora_ciclica(12 * 60)
    assert (sin_12, cos_12) == pytest.approx((0.0, -1.0))


def test_los_minutos_dan_la_vuelta():
    assert hora_ciclica(1440) == pytest.approx(hora_ciclica(0))
    assert hora_ciclica(1500) == pytest.approx(hora_ciclica(60))


# --- El contexto de comida -------------------------------------------------------------------


def test_sin_contexto_tiene_su_propio_codigo_y_no_es_cero():
    """«Sin contexto» es un valor, no la ausencia de uno.

    Si `None` compartiera código con `en_ayunas`, el modelo aprendería que la mitad de los
    registros nocturnos son en ayunas. El contexto es opcional en la app a propósito: forzarlo
    durante una hipoglucemia de madrugada sería peor, y una etiqueta inventada es peor que
    ninguna.
    """
    assert codigo_contexto(None) == 5.0
    assert codigo_contexto("en_ayunas") == 0.0
    assert CODIGOS_CONTEXTO[None] != CODIGOS_CONTEXTO["en_ayunas"]


def test_los_codigos_del_contexto_son_estos():
    """Cambiar un número de aquí obliga a reentrenar: el artefacto aprendió los códigos."""
    assert CODIGOS_CONTEXTO == {
        "en_ayunas": 0,
        "antes_de_comer": 1,
        "dos_horas_despues": 2,
        "antes_de_dormir": 3,
        "otro": 4,
        None: 5,
    }


def test_un_contexto_desconocido_revienta():
    """Significa que el enum cambió de un lado y no del otro."""
    with pytest.raises(ValueError, match="contexto de comida desconocido"):
        codigo_contexto("despues_de_correr")


# --- El vector, que es donde se desplazan las columnas ---------------------------------------


def _fila_densa() -> dict[str, float]:
    return {nombre: float(i) for i, nombre in enumerate(CARACTERISTICAS_DENSAS)}


def test_el_vector_sale_en_el_orden_de_la_tupla():
    fila = _fila_densa()
    assert a_vector(fila, "denso") == [float(i) for i in range(len(CARACTERISTICAS_DENSAS))]


def test_el_orden_del_diccionario_no_cambia_el_vector():
    """Es la garantía de verdad: el diccionario puede venir como sea.

    Las filas llegarán de un `DataFrame`, de un JSON o de una consulta, y ninguno promete
    orden. El orden del vector lo decide la tupla y nada más.
    """
    fila = _fila_densa()
    al_reves = dict(reversed(list(fila.items())))
    assert a_vector(al_reves, "denso") == a_vector(fila, "denso")


def test_una_llave_que_falta_revienta_en_lugar_de_rellenarse():
    """Rellenar con cero desplazaría la columna, que es el error que esto evita."""
    fila = _fila_densa()
    del fila["iob"]
    with pytest.raises(ValueError, match="faltan"):
        a_vector(fila, "denso")


def test_una_llave_de_mas_revienta():
    """Una característica nueva que el backend no conoce no puede entrar por la puerta de atrás."""
    fila = _fila_densa() | {"actividad": 7.0}
    with pytest.raises(ValueError, match="sobran"):
        a_vector(fila, "denso")


def test_un_valor_que_falta_es_nan_y_eso_si_es_legitimo():
    """Que falte el **valor** es información; que falte la **llave** es un error.

    Un rezago de 60 minutos que no existe porque la serie empieza ahí no es una glucosa de
    cero: XGBoost aprende para qué lado mandar un `NaN`, y el cero le arrastra el promedio.
    """
    fila = _fila_densa() | {"rezago_60": FALTA}
    vector = a_vector(fila, "denso")
    assert math.isnan(vector[CARACTERISTICAS_DENSAS.index("rezago_60")])
    assert math.isnan(FALTA)


def test_el_juego_disperso_tambien_se_arma():
    fila = {nombre: 1.0 for nombre in CARACTERISTICAS_DISPERSAS}
    assert len(a_vector(fila, "disperso")) == len(CARACTERISTICAS_DISPERSAS)


def test_un_juego_desconocido_revienta():
    with pytest.raises(ValueError, match="juego desconocido"):
        a_vector({}, "semidenso")


def test_la_matriz_falla_en_la_fila_mala_y_no_a_medias():
    buena = _fila_densa()
    mala = _fila_densa()
    del mala["cob"]
    with pytest.raises(ValueError):
        a_matriz([buena, buena, mala], "denso")


def test_el_manifiesto_es_una_copia_y_no_la_tupla():
    """`GET /predict` compara contra esta lista; ordenarla en el sitio movería el contrato."""
    manifiesto = manifiesto_de_caracteristicas("denso")
    assert manifiesto == list(CARACTERISTICAS_DENSAS)
    manifiesto.sort()
    assert manifiesto_de_caracteristicas("denso") == list(CARACTERISTICAS_DENSAS)
