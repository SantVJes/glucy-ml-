"""El recorrido del bosque que lleva dentro un `.mlmodel`, con nodos armados a mano.

**Biblioteca estándar y nada que instalar**, así que corren en la integración continua igual
que el resto. Lo que convierte de verdad está en `test_export_coreml.py`, que se salta cuando
no hay xgboost ni coremltools.
"""

import math

import pytest

from src.export.arbol import (
    HOJA,
    MAYOR,
    MAYOR_O_IGUAL,
    MENOR,
    MENOR_O_IGUAL,
    Ensamble,
    Nodo,
)

NADA = (0, 0.0, 0, 0, False, 0.0)


def hoja(valor: float) -> Nodo:
    return Nodo(HOJA, 0, 0.0, 0, 0, False, valor)


def corte(
    comportamiento: int,
    umbral: float,
    verdadero: int,
    falso: int,
    *,
    caracteristica: int = 0,
    faltante_al_verdadero: bool = False,
) -> Nodo:
    return Nodo(
        comportamiento, caracteristica, umbral, verdadero, falso, faltante_al_verdadero, 0.0
    )


def un_arbol(comportamiento: int = MENOR, *, faltante_al_verdadero: bool = False) -> Ensamble:
    """Un árbol de un corte: si la condición se cumple, 10; si no, 20. Base 100."""
    return Ensamble(
        nodos={
            (0, 0): corte(comportamiento, 5.0, 1, 2, faltante_al_verdadero=faltante_al_verdadero),
            (0, 1): hoja(10.0),
            (0, 2): hoja(20.0),
        },
        arboles=(0,),
        base=100.0,
    )


# El recorrido del bosque -----------------------------------------------------


@pytest.mark.parametrize(
    ("comportamiento", "valor", "esperado"),
    [
        (MENOR, 4.0, 110.0),
        (MENOR, 5.0, 120.0),  # el umbral exacto NO cumple «menor que»
        (MENOR_O_IGUAL, 5.0, 110.0),  # y sí cumple «menor o igual»
        (MAYOR, 6.0, 110.0),
        (MAYOR, 5.0, 120.0),
        (MAYOR_O_IGUAL, 5.0, 110.0),
    ],
)
def test_cada_comportamiento_de_corte_manda_la_fila_al_lado_que_le_toca(
    comportamiento, valor, esperado
):
    """Confundir «menor» con «mayor» invierte cada corte y el modelo sigue prediciendo."""
    assert un_arbol(comportamiento).predecir([valor]) == esperado


def test_los_numeros_del_comportamiento_no_son_los_intuitivos():
    """Los «menor que» van antes que los «mayor que», al revés de lo que uno supondría.

    Está escrito como prueba porque es el error que esta comprobación encontró primero, y
    estaba en el comprobador: con el orden al revés, la conversión parecía estar mal por 90
    mg/dL de media cuando lo que estaba mal era quien la medía.
    """
    assert (MENOR_O_IGUAL, MENOR, MAYOR_O_IGUAL, MAYOR, HOJA) == (0, 1, 2, 3, 6)


def test_el_valor_que_falta_va_por_donde_dice_la_marca_y_no_por_la_comparacion():
    """Con `NaN`, una comparación IEEE siempre da falso; la marca puede decir lo contrario.

    Es la diferencia que importa en modo sensor con huecos en la serie: la regla del proyecto
    es que lo que falta es `NaN` y nunca cero, y XGBoost aprende para qué lado mandarlo.
    """
    al_verdadero = un_arbol(MENOR, faltante_al_verdadero=True)
    assert al_verdadero.predecir([math.nan]) == 110.0
    assert al_verdadero.predecir([math.nan], respetar_faltante=False) == 120.0

    al_falso = un_arbol(MENOR, faltante_al_verdadero=False)
    assert al_falso.predecir([math.nan]) == 120.0
    # Sin marca hacia el hijo verdadero, un sistema que ignore la marca acierta por su cuenta.
    assert al_falso.predecir([math.nan], respetar_faltante=False) == 120.0


def test_el_bosque_suma_los_arboles_y_el_valor_de_base():
    ensamble = Ensamble(
        nodos={
            (0, 0): hoja(1.0),
            (1, 0): hoja(2.0),
            (2, 0): corte(MENOR, 5.0, 1, 2),
            (2, 1): hoja(4.0),
            (2, 2): hoja(8.0),
        },
        arboles=(0, 1, 2),
        base=100.0,
    )
    assert ensamble.predecir([1.0]) == 107.0
    assert ensamble.predecir([9.0]) == 111.0


def test_cuenta_los_cortes_y_los_que_mandan_el_faltante_al_hijo_verdadero():
    ensamble = Ensamble(
        nodos={
            (0, 0): corte(MENOR, 5.0, 1, 2, faltante_al_verdadero=True),
            (0, 1): corte(MENOR, 2.0, 3, 4),
            (0, 2): hoja(1.0),
            (0, 3): hoja(2.0),
            (0, 4): hoja(3.0),
        },
        arboles=(0,),
        base=0.0,
    )
    assert ensamble.cortes == 2
    assert ensamble.cortes_con_faltante_al_verdadero == 1
