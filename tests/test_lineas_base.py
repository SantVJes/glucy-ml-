"""Pruebas de las líneas base.

Lo que de verdad comprueban es el alineamiento: que la predicción del punto `i` se compare
contra `serie[i + horizonte]` y no contra `serie[i]`. Un error de un solo índice aquí mejora
o empeora todas las métricas del reporte y no se ve en ninguna parte.
"""

import math

import pytest

from src.eval.baselines import extrapolacion_lineal, pares_con_reales, persistencia
from src.eval.metrics import rmse

SERIE = [100.0, 105.0, 112.0, 120.0, 118.0, 115.0, 110.0]


# --- Persistencia ---------------------------------------------------------------------------


def test_persistencia_predice_el_valor_de_ahora():
    assert persistencia(SERIE, 3) == [100.0, 105.0, 112.0, 120.0]


def test_persistencia_da_una_prediccion_por_cada_real_disponible():
    """`len(serie) - horizonte`: los últimos puntos no tienen con qué compararse."""
    for pasos in (1, 2, 3, 6):
        assert len(persistencia(SERIE, pasos)) == len(SERIE) - pasos


def test_persistencia_da_lo_mismo_que_la_referencia_del_backend():
    """`linea_base_persistencia(serie, 3)` de `referencia/reglas_clinicas.py`."""
    assert persistencia([100, 105, 112, 120, 118, 115, 110], 3) == [100, 105, 112, 120]


def test_una_serie_mas_corta_que_el_horizonte_no_da_predicciones():
    assert persistencia([100.0, 105.0], 5) == []
    assert persistencia([100.0, 105.0], 2) == []


def test_un_horizonte_de_cero_pasos_revienta():
    with pytest.raises(ValueError, match="al menos un paso"):
        persistencia(SERIE, 0)


# --- Extrapolación lineal -------------------------------------------------------------------


def test_la_extrapolacion_proyecta_la_pendiente():
    """Una recta de 10 en 10: con pendiente de 10 por paso, a 2 pasos predice 20 más."""
    recta = [float(100 + 10 * i) for i in range(10)]
    predichos = extrapolacion_lineal(recta, 2, pasos_pendiente=3)
    # El punto 3 vale 130 y la pendiente es 10: a dos pasos, 150. Y 150 es el real del punto 5.
    assert predichos[3] == pytest.approx(150.0)
    assert predichos[3] == pytest.approx(recta[3 + 2])


def test_sobre_una_recta_la_extrapolacion_acierta_y_la_persistencia_no():
    """Es el caso que justifica tener las dos: la extrapolación compite en las subidas."""
    recta = [float(100 + 10 * i) for i in range(12)]
    reales_e, pred_e = pares_con_reales(recta, extrapolacion_lineal(recta, 2), 2)
    reales_p, pred_p = pares_con_reales(recta, persistencia(recta, 2), 2)
    assert rmse(reales_e, pred_e) == pytest.approx(0.0)
    assert rmse(reales_p, pred_p) == pytest.approx(20.0)


def test_los_primeros_puntos_salen_nan_y_no_rellenados():
    """Rellenar con el valor actual convertiría esta línea base en la de persistencia.

    Y la haría ver mejor de lo que es justo donde la serie empieza, que es donde menos
    información hay.
    """
    predichos = extrapolacion_lineal(SERIE, 1, pasos_pendiente=3)
    assert all(math.isnan(v) for v in predichos[:3])
    assert not math.isnan(predichos[3])


def test_una_pendiente_de_cero_pasos_revienta():
    with pytest.raises(ValueError, match="al menos un paso"):
        extrapolacion_lineal(SERIE, 1, pasos_pendiente=0)


def test_la_extrapolacion_tambien_respeta_el_largo():
    assert len(extrapolacion_lineal(SERIE, 2)) == len(SERIE) - 2
    assert extrapolacion_lineal([100.0, 105.0], 5) == []


# --- El alineamiento, que es el punto de todo esto ------------------------------------------


def test_el_real_del_punto_i_es_el_de_i_mas_el_horizonte():
    reales, predichos = pares_con_reales(SERIE, persistencia(SERIE, 3), 3)
    assert reales == [120.0, 118.0, 115.0, 110.0]
    assert predichos == [100.0, 105.0, 112.0, 120.0]
    assert len(reales) == len(predichos)


def test_alinear_mal_se_detecta_en_lugar_de_desplazar_las_metricas():
    """Pasarle las predicciones de otro horizonte tiene que reventar, no corregirse solo."""
    with pytest.raises(ValueError, match="alinear mal"):
        pares_con_reales(SERIE, persistencia(SERIE, 1), 3)


def test_los_pares_con_nan_se_descartan_y_el_largo_lo_delata():
    """Una línea base que solo pudo predecir la mitad no se compara de igual a igual.

    De ahí que `Resultado` lleve su `n`: el reporte tiene que decir sobre cuántos pares se
    midió cada línea base.
    """
    reales, predichos = pares_con_reales(SERIE, extrapolacion_lineal(SERIE, 1), 1)
    assert len(reales) == len(SERIE) - 1 - 3
    assert not any(math.isnan(v) for v in predichos)


def test_una_serie_sin_pares_utiles_devuelve_dos_listas_vacias():
    corta = [100.0, 105.0, 110.0, 115.0]
    reales, predichos = pares_con_reales(corta, extrapolacion_lineal(corta, 1), 1)
    assert reales == []
    assert predichos == []
