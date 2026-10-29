"""Pruebas de la rejilla de Clarke.

Las dos primeras **fijan la divergencia** que se encontró al cotejar la implementación del
backend contra el port público de la versión en MATLAB. Si alguien vuelve a la formulación
laxa, estas dos pruebas lo dicen en lugar de que el A+B del reporte suba sin explicación.
"""

import pytest

from src.eval.clarke import HIPER, HIPO, ZONAS, porcentaje_ab, reparto, zona, zonas_de

# --- Las dos divergencias verificadas, que son la razón de ser de este archivo ---------------


def test_zona_c_llega_hasta_290_y_no_se_corta_en_180():
    """Real 200 con predicho 320 es **C**, no B.

    La implementación del backend topa la zona C en 180 mg/dL, y con eso 28 105 pares de
    reales entre 181 y 290 que el port público llama C caen en B. B cuenta para el A+B que el
    proyecto promete ≥ 95 %, así que el corte laxo infla justo la métrica principal.
    """
    assert zona(200, 320) == "C"


def test_predicho_exactamente_en_180_con_real_alto_es_d():
    """Real 300 con predicho exactamente 180 es **D**, no B.

    El límite de la zona D es inclusivo: `70 <= predicho <= 180`. Con la comparación estricta
    del backend, 361 pares caen en B. Es un caso de borde y va escrito porque un `<` por un
    `<=` no se ve leyendo el código.
    """
    assert zona(300, 180) == "D"


def test_una_hipoglucemia_no_detectada_es_d_y_no_b():
    """Real 60 con predicho 120 es **D**, no B.

    Es la tercera familia del cotejo y la única donde el puerto público es el indulgente: su
    condición para la zona D baja pide `p >= (7/5)*real + 110`, que con un real de 60 son 194,
    y arriba de 180 ya manda la zona E, así que esa rama nunca se activa y le salen B 1 224
    pares con reales de 59 a 70.

    Pero un real de 60 predicho como 120 es literalmente «no detectar una hipoglucemia», que
    es la definición de la zona D y **el evento central de Glucy**. La referencia del backend
    aquí tiene razón y este repositorio la sigue.
    """
    assert zona(60, 120) == "D"
    assert zona(59, 71) == "D"


# --- Las cinco zonas, cada una con un caso claro --------------------------------------------


def test_zona_a_por_tolerancia_del_veinte_por_ciento():
    # 100 ± 20 es zona A: clínicamente el mismo dato.
    assert zona(100, 115) == "A"
    assert zona(100, 85) == "A"
    assert zona(100, 121) != "A"


def test_zona_a_cuando_los_dos_estan_en_hipoglucemia():
    """Los dos por debajo de 70 es A aunque el error relativo sea grande.

    50 predicho como 65 es 30 % de error y sin embargo es la misma decisión clínica: hay
    hipoglucemia y hay que actuar. Es el caso que la tolerancia del 20 % sola clasificaría mal
    y el motivo de que la rejilla exista en lugar de un error porcentual.
    """
    assert zona(50, 65) == "A"
    assert zona(HIPO, HIPO) == "A"


def test_zona_e_es_el_error_que_invierte_el_tratamiento():
    # Hipoglucemia real leída como hiperglucemia, y al revés: el error peligroso.
    assert zona(50, 200) == "E"
    assert zona(250, 50) == "E"


def test_zona_b_es_el_error_grande_pero_sin_consecuencia():
    assert zona(150, 200) == "B"


def test_zona_d_es_no_detectar_lo_que_habia():
    """Real 300 leído como 150: no se detecta la hiperglucemia, pero no se trata al revés."""
    assert zona(300, 150) == "D"


def test_zona_d_tambien_del_lado_bajo():
    """Real 50 leído como 120: una hipoglucemia que pasa desapercibida.

    Es el caso que más importa para Glucy: el sistema promete anticipar hipoglucemias, y una
    predicción así es exactamente la falla que la sensibilidad mide.
    """
    assert zona(50, 120) == "D"


# --- Invariantes del módulo ------------------------------------------------------------------


def test_toda_pareja_cae_en_una_de_las_cinco_zonas():
    """Ninguna pareja válida se queda sin zona, barriendo la rejilla completa.

    Un solo caso de prueba con el barrido dentro y no `parametrize`: son 337 561 parejas, y
    337 561 casos de prueba harían ilegible el conteo de la suite y lento el reporte de pytest.
    """
    sin_zona = [
        (real, predicho)
        for real in range(20, 601)
        for predicho in range(20, 601)
        if zona(real, predicho) not in ZONAS
    ]
    assert sin_zona == []


def test_la_prediccion_perfecta_es_toda_a():
    serie = [70, 100, 180, 250, 400]
    assert reparto(serie, serie) == {"A": 100.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0}
    assert porcentaje_ab(serie, serie) == 100.0


def test_el_reparto_siempre_trae_las_cinco_llaves():
    """Aunque ninguna pareja caiga en una zona, su llave está con 0.0.

    Un diccionario al que le falta la llave «E» obliga a quien escribe el reporte a usar
    `.get("E", 0)` y es un cero que se lee como «no se midió». Las cinco siempre.
    """
    r = reparto([100], [100])
    assert sorted(r) == ["A", "B", "C", "D", "E"]
    assert r["E"] == 0.0


def test_el_reparto_suma_cien():
    reales = [45, 70, 100, 150, 200, 300, 400]
    predichos = [60, 200, 115, 210, 320, 180, 100]
    r = reparto(reales, predichos)
    assert sum(r.values()) == pytest.approx(100.0)


def test_porcentaje_ab_cuenta_a_y_b_juntas():
    reales = [100, 100, 50, 300]
    predichos = [100, 200, 200, 180]  # A, B, E, D
    assert porcentaje_ab(reales, predichos) == pytest.approx(50.0)


def test_sin_parejas_revienta_en_lugar_de_devolver_cien():
    """Un conjunto vacío no es una predicción perfecta.

    Devolver 100 % de A+B sobre cero parejas pondría la mejor cifra posible en el reporte
    precisamente cuando no se midió nada.
    """
    with pytest.raises(ValueError):
        zonas_de([], [])
    with pytest.raises(ValueError):
        porcentaje_ab([], [])


def test_largos_distintos_revientan():
    """Desalinear reales y predichos cambia todas las métricas sin que nada falle."""
    with pytest.raises(ValueError):
        zonas_de([100, 120], [100])


def test_los_umbrales_son_los_del_proyecto():
    """70 y 180, los mismos que el perfil y los mismos que el backend."""
    assert HIPO == 70
    assert HIPER == 180
