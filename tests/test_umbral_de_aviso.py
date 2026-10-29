"""Pruebas del umbral de aviso.

Este módulo existe para corregir una comparación que estaba mal hecha, así que lo que las
pruebas tienen que fijar no es que los porcentajes se calculen —eso ya lo hacía `metrics.py`—
sino las tres cosas que hacen que la corrección sea una corrección:

1. Que el **umbral clínico no se mueva** cuando se mueve el de aviso. Si se movieran juntos,
   se estaría cambiando la definición de hipoglucemia para que el modelo la acierte.
2. Que un modelo que **encoge hacia la media** salga con menos sensibilidad al mismo umbral y
   la recupere al subirlo. Es el mecanismo entero, reproducido con números a mano.
3. Que «cumple las metas» exija **las dos a la vez**, porque cada una por separado se cumple
   con un umbral absurdo.
"""

import pytest

from src.eval.umbral_de_aviso import (
    UMBRALES_POR_OMISION,
    comparar,
    curva,
    punto,
    sensibilidad_a_falsas_igualadas,
    umbral_que_cumple_las_metas,
)

# Ocho pares a mano. Tres episodios reales de hipoglucemia (60, 65, 68) y cinco que no lo son.
REALES = [60.0, 65.0, 68.0, 72.0, 90.0, 100.0, 120.0, 150.0]

# Un modelo que encoge hacia 100: el 60 real lo predice en 68, así que con aviso a 70 lo caza,
# pero el 65 lo predice en 72 y se le escapa.
ENCOGIDO = [68.0, 72.0, 74.4, 77.6, 92.0, 100.0, 116.0, 140.0]

# La persistencia del ejemplo: copia el valor real, así que caza los tres episodios al umbral
# clínico y no se equivoca ninguna vez.
COPIA = list(REALES)


def test_el_umbral_clinico_no_se_mueve_cuando_se_mueve_el_de_aviso():
    """Es la línea que separa medir mejor de hacer trampa.

    Los episodios que cuentan son siempre los que de verdad bajaron de 70. Si al subir el
    umbral de aviso subiera también el clínico, el denominador crecería con valores que no son
    hipoglucemias y la sensibilidad saldría inflada sin que ningún número se viera raro.
    """
    bajo = punto(REALES, ENCOGIDO, 70)
    alto = punto(REALES, ENCOGIDO, 100)
    assert bajo.n_episodios == 3
    assert alto.n_episodios == 3, "el umbral de aviso cambió el denominador clínico"


def test_un_modelo_que_encoge_caza_menos_al_mismo_umbral_y_mas_al_subirlo():
    """El mecanismo entero, que es lo que motivó el módulo.

    Con aviso a 70 el modelo encogido caza uno de tres y la copia caza tres de tres, y de ahí
    se concluiría que el modelo es peor. Subiendo el aviso a 75 el modelo caza los tres: no
    era peor, estaba medido en otro punto de la curva.
    """
    assert punto(REALES, ENCOGIDO, 70).sensibilidad == pytest.approx(100 / 3)
    assert punto(REALES, COPIA, 70).sensibilidad == pytest.approx(100.0)
    assert punto(REALES, ENCOGIDO, 75).sensibilidad == pytest.approx(100.0)


def test_subir_el_umbral_nunca_baja_la_sensibilidad():
    """La curva es monótona en sensibilidad, por construcción: avisar antes solo puede cazar
    más. Si alguna vez dejara de serlo, el cálculo estaría roto."""
    sensibilidades = [p.sensibilidad for p in curva(REALES, ENCOGIDO)]
    assert sensibilidades == sorted(sensibilidades)


def test_la_curva_recorre_los_umbrales_en_orden_y_completa():
    puntos = curva(REALES, ENCOGIDO)
    assert [p.umbral for p in puntos] == sorted(UMBRALES_POR_OMISION)


def test_igualar_falsas_alarmas_es_la_comparacion_que_vale():
    """Con la molestia igualada, el modelo encogido caza lo mismo que la copia.

    La copia no se equivoca nunca en este ejemplo (0 % de falsas), así que igualar su tasa
    obliga al modelo a quedarse en un umbral donde tampoco se equivoca. Ahí caza los tres.
    """
    base = punto(REALES, COPIA, 70)
    assert base.falsas == pytest.approx(0.0)
    igualado = sensibilidad_a_falsas_igualadas(REALES, ENCOGIDO, base.falsas)
    assert igualado is not None
    assert igualado.sensibilidad == pytest.approx(100.0)
    assert igualado.falsas <= base.falsas


def test_sin_ningun_punto_por_debajo_del_objetivo_devuelve_none():
    """Devolver el punto menos malo escondería que el modelo no tiene un punto aceptable."""
    assert sensibilidad_a_falsas_igualadas(REALES, ENCOGIDO, -1.0) is None


def test_cumplir_las_metas_exige_las_dos_a_la_vez():
    """Cada meta por separado se cumple con un umbral absurdo, y por eso van juntas.

    Un umbral muy alto caza casi todo a base de avisar casi siempre; uno muy bajo casi no se
    equivoca porque casi no avisa. Solo el par significa algo.
    """
    muy_alto = punto(REALES, ENCOGIDO, 200)
    assert muy_alto.sensibilidad == pytest.approx(100.0)
    assert muy_alto.falsas > 15.0
    assert not muy_alto.cumple_las_metas


def test_el_umbral_elegido_es_el_mas_bajo_de_los_que_cumplen():
    """Con las metas cumplidas, lo que queda por optimizar es no molestar: una app que avisa de
    más se apaga sola, y eso es criterio de aceptación."""
    elegido = umbral_que_cumple_las_metas(REALES, ENCOGIDO)
    assert elegido is not None
    assert elegido.cumple_las_metas
    cumplen = [p.umbral for p in curva(REALES, ENCOGIDO) if p.cumple_las_metas]
    assert elegido.umbral == min(cumplen)


def test_sin_episodios_reales_revienta_en_lugar_de_decir_cien_por_ciento():
    """Un conjunto sin hipoglucemias no demuestra que el modelo las detecte, y un 100 % en esa
    casilla del reporte sería la cifra más engañosa del documento."""
    sanos = [100.0, 120.0, 150.0]
    with pytest.raises(ValueError, match="no hay episodios reales"):
        punto(sanos, sanos, 70)


def test_sin_ninguna_alerta_las_falsas_son_cero_y_la_sensibilidad_lo_delata():
    """Cero falsas porque no se emitió ninguna es honesto; la sensibilidad en cero es lo que
    enseña que no se emitió ninguna."""
    p = punto(REALES, ENCOGIDO, 55)
    assert p.n_alertas == 0
    assert p.falsas == pytest.approx(0.0)
    assert p.sensibilidad == pytest.approx(0.0)


def test_comparar_dice_si_el_problema_es_el_umbral_o_son_las_caracteristicas():
    """El texto es el resultado que se lee, así que su conclusión también se prueba."""
    texto = comparar(REALES, ENCOGIDO, COPIA)
    assert "línea base" in texto
    assert "igualando esas falsas" in texto
    assert "cazan lo mismo" in texto or "puntos MÁS" in texto or "puntos MENOS" in texto


def test_un_modelo_peor_en_todos_sus_puntos_se_dice_tal_cual():
    """Si ni el umbral más bajo baja de la tasa de la línea base, el texto no puede sugerir que
    moviendo el umbral se arregla."""
    reales = [60.0, 65.0, 100.0, 120.0, 150.0]
    # Predice bajo para todo: siempre avisa, así que siempre se equivoca mucho.
    siempre_bajo = [50.0] * 5
    texto = comparar(reales, siempre_bajo, list(reales))
    assert "ningún umbral" in texto


def test_el_entrenamiento_imprime_la_comparacion_y_aguanta_una_corrida_sin_predicciones(capsys):
    """La comparación se imprime en cada corrida, así que el cableado también se prueba.

    El segundo caso es una `Corrida` sin predicciones: el campo tiene valor por omisión, y una
    corrida construida sin él no puede reventar el entrenamiento entero por no poder imprimir
    un extra.
    """
    pytest.importorskip(
        "xgboost", reason="src.train.comun arrastra las dependencias de entrenamiento"
    )
    from src.train.comun import _imprimir_umbral_de_aviso
    from src.train.entrenar import Corrida, Prediccion, _medir

    del_modelo = tuple(
        Prediccion(sujeto="s1", real=r, predicho=p) for r, p in zip(REALES, ENCOGIDO, strict=True)
    )
    de_la_base = tuple(Prediccion(sujeto="s1", real=r, predicho=r) for r in REALES)
    comun = dict(
        modelo="m1",
        horizonte_min=30,
        n_entrenamiento=len(REALES),
        resultado=_medir(del_modelo, 30, 70),
        por_sujeto={},
        sujetos=("s1",),
    )

    _imprimir_umbral_de_aviso(
        Corrida(
            **comun,
            lineas_base={"persistencia": _medir(de_la_base, 30, 70)},
            predicciones=del_modelo,
            predicciones_base={"persistencia": de_la_base},
        ),
        70,
    )
    salida = capsys.readouterr().out
    assert "igualando la tasa de falsas alarmas" in salida
    assert "aviso < 75 mg/dL" in salida

    _imprimir_umbral_de_aviso(Corrida(**comun, lineas_base={}), 70)
    assert capsys.readouterr().out == ""
