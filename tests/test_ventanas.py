"""Pruebas de las ventanas de entrenamiento.

La primera es la que más vale: **que una fila no mire hacia adelante**. Se comprueba recortando
la serie después del punto que se predice y exigiendo exactamente el mismo vector. Si alguna
característica se colara desde el futuro, el vector cambiaría.
"""

import math
from datetime import datetime, timedelta

import pytest

from src.features.features import CARACTERISTICAS_DENSAS, CARACTERISTICAS_DISPERSAS, a_vector
from src.features.ventanas import (
    TOLERANCIA_OBJETIVO_MIN,
    caracteristicas_densas,
    caracteristicas_dispersas,
    cob,
    filas_densas,
    filas_pico,
    iob,
)
from src.ingest.ohiot1dm import Evento, Lectura

ARRANQUE = datetime(2026, 3, 1, 8, 0)


def serie(valores, *, paso_min=5, sujeto="S00", del_sensor=True):
    return [
        Lectura(
            sujeto=sujeto,
            ts=ARRANQUE + timedelta(minutes=i * paso_min),
            mg_dl=float(v),
            del_sensor=del_sensor,
        )
        for i, v in enumerate(valores)
    ]


def evento(minutos, valor, seccion="meal", sujeto="S00"):
    return Evento(
        sujeto=sujeto,
        seccion=seccion,
        ts=ARRANQUE + timedelta(minutes=minutos),
        valor=valor,
    )


# --- No mirar hacia adelante -----------------------------------------------------------------


def test_una_fila_no_cambia_si_se_recorta_el_futuro():
    """La garantía central. En producción no existe el futuro: a las 14:00 no hay las 14:05."""
    larga = serie(range(100, 140))
    corta = larga[:21]  # hasta el punto 20 inclusive
    comidas = [evento(30, 40.0)]
    dosis = [evento(35, 4.0, "bolus")]

    con_futuro = caracteristicas_densas(larga[:21], comidas, dosis)
    sin_futuro = caracteristicas_densas(corta, comidas, dosis)
    assert con_futuro == sin_futuro


def test_las_filas_se_arman_solo_con_la_historia_hasta_ese_punto():
    """La misma garantía, pero a través de `filas_densas`, que es quien recorta."""
    larga = serie(range(100, 160))
    filas_largas = filas_densas(larga, [], [], horizonte_min=30)
    filas_cortas = filas_densas(larga[:30], [], [], horizonte_min=30)
    comunes = min(len(filas_largas), len(filas_cortas))
    assert comunes > 5
    for a, b in zip(filas_largas[:comunes], filas_cortas[:comunes], strict=True):
        assert a.caracteristicas == b.caracteristicas


# --- Rezagos por tiempo, no por posición -----------------------------------------------------


def test_los_rezagos_se_buscan_por_tiempo_y_no_por_posicion():
    """Con un hueco, el rezago de 30 minutos no es «seis filas atrás».

    HealthKit no garantiza una serie regular y OhioT1DM trae huecos. Buscar por posición da un
    valor de otro momento, con el nombre de este.
    """
    lecturas = serie([100, 110, 120, 130, 140, 150, 160])
    # Se quita la cuarta lectura: queda un hueco de 10 minutos.
    con_hueco = lecturas[:3] + lecturas[4:]
    fila = caracteristicas_densas(con_hueco, [], [])
    # El último punto está a 30 min del inicio... con el hueco, a 30 min atrás ya no hay nada
    # exactamente, así que el rezago_30 tiene que ser el valor de ese instante si existe.
    assert fila["rezago_5"] == 150.0
    assert fila["rezago_30"] == 100.0


def test_un_rezago_que_no_existe_es_nan_y_no_cero():
    """Una serie que empieza no tiene una hora de pasado, y eso no es una glucosa de cero."""
    fila = caracteristicas_densas(serie([100, 110, 120]), [], [])
    assert math.isnan(fila["rezago_60"])
    assert math.isnan(fila["rezago_30"])
    assert fila["rezago_5"] == 110.0


def test_las_derivadas_son_nan_cuando_faltan_sus_rezagos():
    """Suponer «cambio cero» al arrancar enseñaría al modelo a predecir quietud sin saber."""
    fila = caracteristicas_densas(serie([100]), [], [])
    assert math.isnan(fila["derivada_1"])
    assert math.isnan(fila["derivada_2"])


def test_la_primera_derivada_es_mg_dl_por_minuto():
    fila = caracteristicas_densas(serie([100, 110]), [], [])
    assert fila["derivada_1"] == pytest.approx(2.0)  # 10 mg/dL en 5 min


def test_las_medias_moviles_solo_usan_lo_que_cae_en_la_ventana():
    lecturas = serie(range(100, 120))  # 20 lecturas cada 5 min = 95 minutos
    fila = caracteristicas_densas(lecturas, [], [])
    # La media de 30 minutos toma los últimos 7 puntos (0, 5, ..., 30 minutos atrás).
    esperada = sum(range(113, 120)) / 7
    assert fila["media_30"] == pytest.approx(esperada)


# --- El objetivo -----------------------------------------------------------------------------


def test_el_objetivo_es_la_lectura_del_horizonte():
    filas = filas_densas(serie(range(100, 120)), [], [], horizonte_min=30)
    primera = filas[0]
    assert primera.caracteristicas["mg_dl"] == 100.0
    assert primera.objetivo == 106.0  # 30 min después son 6 pasos de 5 min


def test_una_fila_sin_objetivo_cerca_se_descarta():
    """Un objetivo a 50 minutos etiquetado como «a 30» es ruido con apariencia de dato."""
    # Dos bloques separados por dos horas: las filas del primero no tienen objetivo a 30 min.
    primero = serie([100, 105, 110])
    segundo = [
        Lectura(
            sujeto="S00",
            ts=ARRANQUE + timedelta(minutes=120 + i * 5),
            mg_dl=200.0 + i,
            del_sensor=True,
        )
        for i in range(3)
    ]
    filas = filas_densas(primero + segundo, [], [], horizonte_min=30)
    assert filas == []


def test_el_objetivo_acepta_un_desfase_dentro_de_la_tolerancia():
    lecturas = serie([100, 110, 120, 130, 140, 150, 160])
    filas = filas_densas(lecturas, [], [], horizonte_min=28)
    assert filas
    assert TOLERANCIA_OBJETIVO_MIN >= 2


def test_los_pinchazos_no_entran_al_juego_denso_por_omision():
    """Tres pinchazos sueltos no son una serie: sus rezagos serían basura."""
    del_sensor = serie([100, 105, 110, 115, 120, 125, 130])
    pinchazo = [
        Lectura(sujeto="S00", ts=ARRANQUE + timedelta(minutes=7), mg_dl=95.0, del_sensor=False)
    ]
    todas = sorted(del_sensor + pinchazo, key=lambda f: f.ts)
    filas = filas_densas(todas, [], [], horizonte_min=30)
    assert all(f.caracteristicas["mg_dl"] != 95.0 for f in filas)


def test_un_horizonte_de_cero_revienta():
    with pytest.raises(ValueError, match="al menos un minuto"):
        filas_densas(serie([100, 110]), [], [], horizonte_min=0)


# --- COB e IOB, que tienen que dar lo mismo que el backend ----------------------------------


def test_cob_60_g_con_180_min_a_los_90_min_son_30_g():
    """El caso P-11, el que más se cita en todo el proyecto.

    Absorción lineal, no una curva de digestión (D-8). Los mismos números que da
    `referencia/reglas_clinicas.py` del backend y que tiene que dar el Swift.
    """
    comida = evento(0, 60.0)
    ahora = (ARRANQUE + timedelta(minutes=90)).timestamp()
    assert cob([comida], ahora, 180) == pytest.approx(30.0)


def test_cob_es_cero_pasado_el_tiempo_de_absorcion():
    comida = evento(0, 60.0)
    assert cob([comida], (ARRANQUE + timedelta(minutes=180)).timestamp(), 180) == 0.0
    assert cob([comida], (ARRANQUE + timedelta(minutes=400)).timestamp(), 180) == 0.0


def test_cob_no_cuenta_una_comida_que_todavia_no_ocurrio():
    """Mirar hacia adelante, otra vez: una comida futura no puede estar activa."""
    futura = evento(60, 50.0)
    assert cob([futura], ARRANQUE.timestamp(), 180) == 0.0


def test_cob_suma_varias_comidas():
    comidas = [evento(0, 60.0), evento(90, 30.0)]
    ahora = (ARRANQUE + timedelta(minutes=90)).timestamp()
    assert cob(comidas, ahora, 180) == pytest.approx(60.0)  # 30 de la primera + 30 de la segunda


def test_cob_ignora_una_comida_sin_gramos():
    """Una comida registrada sin carbohidratos no aporta; tampoco revienta."""
    assert cob([evento(0, None)], ARRANQUE.timestamp(), 180) == 0.0


def test_iob_baja_hasta_cero_al_terminar_la_duracion():
    dosis = [evento(0, 10.0, "bolus")]
    inicio = iob(dosis, ARRANQUE.timestamp())
    media = iob(dosis, (ARRANQUE + timedelta(hours=2.5)).timestamp())
    final = iob(dosis, (ARRANQUE + timedelta(hours=5)).timestamp())
    assert inicio == pytest.approx(10.0)
    assert 0 < media < inicio
    assert final == 0.0


def test_iob_no_cuenta_una_dosis_futura():
    assert iob([evento(60, 5.0, "bolus")], ARRANQUE.timestamp()) == 0.0


# --- El juego disperso -----------------------------------------------------------------------


def test_el_juego_disperso_no_lleva_ni_un_rezago():
    """Es el modo sin sensor: si llevara rezagos sería inútil justo para quien no tiene sensor."""
    ultima = serie([120])[0]
    fila = caracteristicas_dispersas(ultima, (ARRANQUE + timedelta(minutes=45)).timestamp(), [], [])
    assert not any(nombre.startswith("rezago") for nombre in fila)
    assert "derivada_1" not in fila
    assert a_vector(fila, "disperso")


def test_la_antiguedad_es_la_variable_que_carga_con_la_incertidumbre():
    ultima = serie([120])[0]
    fila = caracteristicas_dispersas(ultima, (ARRANQUE + timedelta(minutes=45)).timestamp(), [], [])
    assert fila["antiguedad_min"] == pytest.approx(45.0)


def test_sin_comidas_ni_dosis_los_campos_quedan_en_nan():
    ultima = serie([120])[0]
    fila = caracteristicas_dispersas(ultima, ARRANQUE.timestamp(), [], [])
    assert math.isnan(fila["carbs_g"])
    assert math.isnan(fila["dosis_ui"])


# --- Las filas de pico (M-3) -----------------------------------------------------------------


def test_una_fila_de_pico_por_comida_registrada():
    lecturas = serie([100, 120, 160, 200, 180, 150, 130, 120, 115])
    comidas = [evento(0, 50.0)]
    filas = filas_pico(lecturas, comidas, [], ventana_min=180, minimo_lecturas=3)
    assert len(filas) == 1
    assert filas[0].pico_mg_dl == 200.0
    assert filas[0].pico_min == pytest.approx(15.0)


def test_no_se_buscan_picos_sin_comida_registrada():
    """D-9 y RF-37: una subida sin comida anotada no se atribuye a una comida."""
    lecturas = serie([100, 150, 220, 180, 140, 120])
    assert filas_pico(lecturas, [], [], minimo_lecturas=2) == []


def test_una_ventana_con_pocas_lecturas_se_descarta():
    """El máximo de dos puntos sueltos no es un pico posprandial."""
    lecturas = serie([100, 150])
    filas = filas_pico(lecturas, [evento(0, 50.0)], [], minimo_lecturas=6)
    assert filas == []


def test_el_pico_usa_el_juego_disperso_completo():
    lecturas = serie([100, 120, 160, 200, 180, 150, 130])
    filas = filas_pico(lecturas, [evento(0, 50.0)], [], minimo_lecturas=3)
    assert sorted(filas[0].caracteristicas) == sorted(CARACTERISTICAS_DISPERSAS)


def test_una_comida_sin_lecturas_previas_se_descarta():
    """Sin una lectura anterior no hay de dónde salir: el juego disperso la necesita."""
    lecturas = serie([100, 120, 160, 200], paso_min=5)
    comida_antes_de_todo = Evento(
        sujeto="S00", seccion="meal", ts=ARRANQUE - timedelta(minutes=30), valor=50.0
    )
    assert filas_pico(lecturas, [comida_antes_de_todo], [], minimo_lecturas=3) == []


# --- El contrato con el backend --------------------------------------------------------------


def test_el_juego_denso_que_sale_cuadra_con_la_tupla_del_contrato():
    """Si `ventanas.py` se olvida de una característica, revienta aquí y no en producción."""
    fila = caracteristicas_densas(serie(range(100, 120)), [evento(10, 40.0)], [])
    assert sorted(fila) == sorted(CARACTERISTICAS_DENSAS)
    assert len(a_vector(fila, "denso")) == len(CARACTERISTICAS_DENSAS)
