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
    # Seis puntos y no siete: la ventana es `(desde, ahora]`, así que la lectura que cae
    # exactamente a 30 minutos queda fuera. Esta prueba fijaba siete, que era la convención
    # equivocada; la del pipeline del backend, que es la autoridad, es esta.
    esperada = sum(range(114, 120)) / 6
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


# --- La traducción de recorrido lineal a búsqueda binaria -----------------------------------
#
# `_mas_cercana` recorría la serie entera y ahora mira los dos vecinos del punto de inserción.
# Una traducción así **no falla, devuelve otra lectura**: el modelo se entrena con otros
# rezagos y las métricas salen distintas sin que nada reviente. Estas tres pruebas fijan lo
# único que la bisección puede cambiar en silencio.


def lectura_en(minutos, mg_dl, sujeto="S00"):
    return Lectura(
        sujeto=sujeto,
        ts=ARRANQUE + timedelta(minutes=minutos),
        mg_dl=float(mg_dl),
        del_sensor=True,
    )


def test_un_empate_exacto_se_queda_con_la_lectura_mas_antigua():
    """Dos lecturas a la misma distancia del instante pedido: gana la de antes.

    Lo hacía el recorrido lineal sin que nadie lo decidiera, porque comparaba con `<` y pasaba
    por la serie en orden. La bisección mira el vecino derecho después del izquierdo, así que
    sin cuidado el empate se habría volteado y los rezagos de las filas con hueco habrían
    cambiado de valor. No es que la antigua sea mejor: es que cambiarlo sin querer habría
    movido números del reporte sin que ninguna prueba lo notara.
    """
    # Historia con un hueco alrededor de los 30 minutos antes de «ahora», que es el minuto 60.
    historia = [
        lectura_en(0, 100),
        lectura_en(27.5, 111),
        lectura_en(32.5, 222),
        lectura_en(60, 150),
    ]
    fila = caracteristicas_densas(historia, [], [])
    assert fila["rezago_30"] == 111.0


def test_el_objetivo_nunca_es_la_propia_fila():
    """Con un horizonte corto, la lectura más cercana a «ahora + horizonte» es «ahora».

    Y «ahora» ya está dentro de la fila, en `mg_dl`: etiquetarla además como objetivo le
    regalaría al modelo la respuesta. Lo que impedía eso era pedir el objetivo sobre
    `usables[i + 1 :]`; ahora lo impide el parámetro `desde`, y la garantía es la misma solo si
    el índice está bien. Un error de uno aquí daría un RMSE casi perfecto y falso.
    """
    filas = filas_densas(serie([100, 120, 140, 160]), [], [], horizonte_min=1)
    assert filas, "con tolerancia de 5 min el objetivo a 1 min tiene que existir"
    for fila, siguiente in zip(filas, [120, 140, 160], strict=False):
        assert fila.objetivo == siguiente
        assert fila.caracteristicas["mg_dl"] != fila.objetivo


def test_una_serie_desordenada_se_rechaza_en_lugar_de_dar_filas_mal():
    """La bisección **exige** orden temporal, y antes era solo una línea de comentario.

    Sobre una serie desordenada una búsqueda binaria no revienta: devuelve la lectura
    equivocada. Así que la comprobación que ya hacía `filas_densas` pasa a ser la condición de
    la que depende la corrección, y por eso ahora tiene prueba.
    """
    desordenada = [lectura_en(0, 100), lectura_en(30, 140), lectura_en(10, 120)]
    with pytest.raises(ValueError, match="ordenada"):
        filas_densas(desordenada, [], [], horizonte_min=30)


# --- El contrato con la referencia clínica -------------------------------------------------
#
# `referencia/reglas_clinicas.py` del backend es la autoridad sobre las reglas clínicas, y el
# pipeline que alimenta al modelo en producción importa de ahí su COB y su IOB. Este
# repositorio no puede importarlo, así que copia la curva y la fija aquí **con valores
# literales**, igual que `test_caracteristicas.py` fija las tuplas del contrato. Es la única
# red posible: un cambio tendría que hacerse a mano y a la vista.
#
# Lo que había antes era una aproximación lineal con un docstring que afirmaba ser «la misma
# forma que usa el backend». No lo era, y el modelo aprendió una columna `iob` que el backend
# nunca le iba a dar. Estas pruebas existen para que no vuelva a pasar en silencio.

# Sacados de `fraccion_insulina_activa` de la referencia, con duración 5 h y pico 75 min.
FRACCION_DE_LA_REFERENCIA = {
    0: 1.0,
    1: 0.999894548,
    15: 0.978824545,
    30: 0.924970186,
    45: 0.850355815,
    60: 0.764005704,
    75: 0.672639890,
    90: 0.581155607,
    120: 0.410579942,
    150: 0.268112684,
    180: 0.158796415,
    240: 0.032924869,
    290: 0.000766432,
    300: 0.0,
    400: 0.0,
}


def test_la_fraccion_activa_es_la_de_la_referencia_valor_por_valor():
    """La prueba que cierra el agujero. Si alguien vuelve a «simplificar» la curva, falla."""
    from src.features.ventanas import fraccion_insulina_activa

    for minutos, esperada in FRACCION_DE_LA_REFERENCIA.items():
        assert fraccion_insulina_activa(float(minutos)) == pytest.approx(esperada, abs=1e-9), (
            f"a los {minutos} min la fracción activa ya no es la de la referencia"
        )


def test_a_las_tres_horas_queda_el_16_por_ciento_y_no_el_40():
    """El número concreto que delató el error, escrito para que se reconozca si reaparece.

    La recta que había antes decía 40 % a los 180 minutos. Con un bolo de 6 UI eso son 1.45 UI
    de insulina que no existen, siempre de más a partir de la primera hora, y entrando a la
    columna `iob` de cada fila de entrenamiento.
    """
    from src.features.ventanas import fraccion_insulina_activa

    assert fraccion_insulina_activa(180) == pytest.approx(0.1588, abs=1e-4)
    recta_vieja = 0.75 * (1 - (180 - 75) / (300 - 75))
    assert recta_vieja == pytest.approx(0.40, abs=1e-9)
    assert abs(fraccion_insulina_activa(180) - recta_vieja) > 0.2


def test_la_fraccion_activa_solo_baja():
    """Una curva de insulina no puede repuntar: lo consumido no vuelve."""
    from src.features.ventanas import fraccion_insulina_activa

    valores = [fraccion_insulina_activa(float(m)) for m in range(0, 301, 5)]
    assert valores == sorted(valores, reverse=True)


def test_una_dosis_futura_no_cuenta_aunque_la_referencia_devuelva_uno():
    """La referencia devuelve 1.0 para minutos negativos, que es lo correcto *para una dosis
    ya puesta*. Armando filas, un negativo es una dosis que todavía no ocurre, y contarla
    sería mirar hacia adelante: la fila tendría información del futuro."""
    assert iob([evento(60, 5.0, "bolus")], ARRANQUE.timestamp()) == 0.0


def test_la_ventana_de_los_promedios_no_incluye_su_borde_inferior():
    """`(desde, ahora]`, como `_media` del pipeline del backend.

    Con el borde inferior incluido, `media_30` sobre una serie de cinco minutos promediaba
    siete lecturas donde producción promedia seis. En una subida eso mueve el promedio unos
    cuantos mg/dL en **cada fila**, y nada falla.
    """
    # Siete lecturas de 5 en 5: la primera cae exactamente a 30 minutos de la última.
    historia = serie([100, 110, 120, 130, 140, 150, 160])
    fila = caracteristicas_densas(historia, [], [])
    seis_ultimas = sum([110, 120, 130, 140, 150, 160]) / 6
    las_siete = sum([100, 110, 120, 130, 140, 150, 160]) / 7
    assert fila["media_30"] == pytest.approx(seis_ultimas)
    assert fila["media_30"] != pytest.approx(las_siete)


def test_la_hora_del_dia_cuenta_los_segundos():
    """El pipeline del backend los cuenta, así que aquí también. Una muestra que no cae en el
    minuto exacto daría otro ángulo que en producción."""
    en_punto = [lectura_en(0, 120)]
    con_segundos = [
        Lectura(
            sujeto="S00",
            ts=ARRANQUE + timedelta(seconds=30),
            mg_dl=120.0,
            del_sensor=True,
        )
    ]
    assert (
        caracteristicas_densas(en_punto, [], [])["hora_sin"]
        != caracteristicas_densas(con_segundos, [], [])["hora_sin"]
    )
