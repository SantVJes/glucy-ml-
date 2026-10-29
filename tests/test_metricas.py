"""Pruebas de las métricas, con los números que da la referencia del backend.

`referencia/reglas_clinicas.py` de `glucy-backend` es el árbitro. Los dos repositorios **no se
pueden importar entre sí**, así que los valores esperados de aquí se calcularon corriendo esa
referencia sobre los mismos pares y están escritos como literales. Si una de estas pruebas
falla, uno de los dos cambió y hay que mirar cuál.

Los pares están elegidos para que los números no sean redondos: un 0.0 o un 100.0 pasan
igual con la fórmula correcta y con una equivocada.
"""

import math

import pytest

from src.eval.metrics import (
    META_CLARKE_AB,
    META_FALSAS_ALARMAS_MAX,
    META_MAE_30_MAX,
    META_MARD_MAX,
    META_RMSE_30_MAX,
    META_SENSIBILIDAD_HIPO,
    Resultado,
    evaluar,
    mae,
    mard,
    rmse,
    sensibilidad_hipoglucemia,
    tasa_falsas_alarmas,
)

# Los mismos pares con los que se corrió la referencia.
REALES = [65, 72, 110, 145, 180, 95, 60, 210]
PREDICHOS = [68, 80, 100, 160, 165, 75, 90, 190]


def test_rmse_da_lo_mismo_que_la_referencia():
    assert rmse(REALES, PREDICHOS) == pytest.approx(17.040393187951974)


def test_mae_da_lo_mismo_que_la_referencia():
    assert mae(REALES, PREDICHOS) == pytest.approx(15.125)


def test_mard_da_lo_mismo_que_la_referencia():
    """En por ciento y dividiendo entre el **real**, no entre el predicho.

    Dividir entre el predicho da un número parecido y distinto, y es el error silencioso que
    este literal atrapa: 15.509 contra 15.9 no se nota mirando.
    """
    assert mard(REALES, PREDICHOS) == pytest.approx(15.509000854962743)


def test_sensibilidad_da_lo_mismo_que_la_referencia():
    # Episodios reales por debajo de 70: 65 y 60. Solo el primero se anticipó.
    assert sensibilidad_hipoglucemia(REALES, PREDICHOS) == pytest.approx(50.0)


def test_falsas_alarmas_da_lo_mismo_que_la_referencia():
    """Tres alertas, dos de ellas sin episodio real detrás: 66.67 %."""
    reales = [100, 60, 120, 65]
    predichos = [65, 55, 68, 90]
    assert tasa_falsas_alarmas(reales, predichos) == pytest.approx(66.66666666666667)
    assert sensibilidad_hipoglucemia(reales, predichos) == pytest.approx(50.0)


# --- Los casos que el reporte no puede permitirse leer mal -----------------------------------


def test_sin_hipoglucemias_reales_revienta_en_lugar_de_dar_cien():
    """Un conjunto sin episodios no demuestra que el modelo los detecte.

    Un 100 % en esa casilla sería la cifra más engañosa del documento, así que la función
    levanta y `evaluar` lo convierte en `None`, que el reporte tiene que escribir como «no se
    midió».
    """
    with pytest.raises(ValueError):
        sensibilidad_hipoglucemia([100, 150], [95, 160])


def test_sin_alertas_las_falsas_alarmas_son_cero_y_eso_si_es_honesto():
    """No se emitió ninguna falsa porque no se emitió ninguna.

    Aquí el cero sí se puede reportar: lo que delata el caso es la sensibilidad, que será
    0 % o levantará. Las dos métricas juntas cuentan la historia completa y por eso ninguna
    se reporta sola.
    """
    assert tasa_falsas_alarmas([100, 60], [110, 90]) == 0.0


def test_el_umbral_de_hipoglucemia_se_puede_mover():
    """El perfil lo mueve entre 50 y 90, y las métricas tienen que seguirlo.

    Con umbral 90, el par (85, 95) pasa a ser un episodio real no anticipado.
    """
    reales = [85, 100]
    predichos = [95, 110]
    with pytest.raises(ValueError):
        sensibilidad_hipoglucemia(reales, predichos, umbral=70)
    assert sensibilidad_hipoglucemia(reales, predichos, umbral=90) == pytest.approx(0.0)


def test_largos_distintos_revientan():
    with pytest.raises(ValueError):
        rmse([100, 120], [100])


def test_sin_pares_revienta():
    for funcion in (rmse, mae, mard):
        with pytest.raises(ValueError):
            funcion([], [])


# --- `evaluar` y el criterio principal -------------------------------------------------------


def test_evaluar_junta_todo_y_conserva_el_tamano_de_la_muestra():
    r = evaluar(REALES, PREDICHOS, horizonte_min=30)
    assert r.n == len(REALES)
    assert r.horizonte_min == 30
    assert r.rmse == pytest.approx(17.040393187951974)
    assert r.mae == pytest.approx(15.125)
    assert sorted(r.clarke_reparto) == ["A", "B", "C", "D", "E"]
    assert sum(r.clarke_reparto.values()) == pytest.approx(100.0)


def test_evaluar_deja_la_sensibilidad_en_none_sin_episodios():
    r = evaluar([100, 150, 200], [105, 160, 190], horizonte_min=30)
    assert r.sensibilidad_hipo is None
    assert not r.cumple_criterio_principal


def test_el_criterio_principal_exige_los_tres():
    """A+B, sensibilidad y falsas alarmas: pasar dos de tres no es pasar."""
    base = {
        "n": 1000,
        "horizonte_min": 30,
        "rmse": 18.0,
        "mae": 13.0,
        "mard": 11.0,
        "clarke_reparto": {"A": 90.0, "B": 6.0, "C": 0.0, "D": 4.0, "E": 0.0},
    }
    bueno = Resultado(clarke_ab=96.0, sensibilidad_hipo=88.0, falsas_alarmas=12.0, **base)
    assert bueno.cumple_criterio_principal

    assert not Resultado(
        clarke_ab=94.9, sensibilidad_hipo=88.0, falsas_alarmas=12.0, **base
    ).cumple_criterio_principal
    assert not Resultado(
        clarke_ab=96.0, sensibilidad_hipo=84.9, falsas_alarmas=12.0, **base
    ).cumple_criterio_principal
    assert not Resultado(
        clarke_ab=96.0, sensibilidad_hipo=88.0, falsas_alarmas=15.1, **base
    ).cumple_criterio_principal


def test_la_sensibilidad_sin_medir_no_cuenta_como_cumplir():
    sin_medir = Resultado(
        n=1000,
        horizonte_min=30,
        rmse=18.0,
        mae=13.0,
        mard=11.0,
        clarke_ab=99.0,
        clarke_reparto={"A": 99.0, "B": 0.0, "C": 0.0, "D": 1.0, "E": 0.0},
        sensibilidad_hipo=None,
        falsas_alarmas=0.0,
    )
    assert not sin_medir.cumple_criterio_principal


def test_las_metas_son_las_del_reporte_y_no_las_de_la_errata():
    """85 % y 15 %, no 80 % y 20 %.

    El documento de la fase F3 en Drive trae los valores de la errata. Esta prueba existe
    para que nadie los «corrija» hacia el documento equivocado.
    """
    assert META_SENSIBILIDAD_HIPO == 85.0
    assert META_FALSAS_ALARMAS_MAX == 15.0
    assert META_CLARKE_AB == 95.0
    assert META_RMSE_30_MAX == 20.0
    assert META_MAE_30_MAX == 15.0
    assert META_MARD_MAX == 12.0


def test_una_prediccion_perfecta_sale_en_cero_de_error():
    serie = [70.0, 100.0, 180.0]
    assert rmse(serie, serie) == 0.0
    assert mae(serie, serie) == 0.0
    assert mard(serie, serie) == 0.0
    assert not math.isnan(mard(serie, serie))
