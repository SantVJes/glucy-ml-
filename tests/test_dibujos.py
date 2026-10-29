"""Pruebas de las figuras del cuaderno de evaluación.

Una figura del reporte no se puede revisar leyendo el código: o se mira, o no se sabe. Lo que
sí se puede afirmar desde una prueba son las cosas que ya salieron mal una vez en este
proyecto, y esas son tres:

1. **Que el dibujo diga lo mismo que el clasificador.** La rejilla se pinta evaluando `zona()`,
   así que no puede discrepar; lo que sí podía, y pasó, es que la letra de una zona cayera
   encima de otra.
2. **Que lo medido salga continuo y lo predicho punteado.** El simulador de la fase 4 los
   dibujó al revés, Altair los repartió por orden alfabético y en una revisión de código no se
   vio. Es la regla de interfaz que el proyecto no puede perder.
3. **Que una figura vacía truene en lugar de salir en blanco.** Un eje sin nada se imprime
   igual de bien que uno con datos.

Saltan enteras si matplotlib no está: la integración continua instala solo pytest y ruff.
"""

import pytest

pytest.importorskip("matplotlib")

from src.eval.clarke import ZONAS, zona  # noqa: E402
from src.eval.dibujos import (  # noqa: E402
    GLUCOSA_MAX,
    _donde_rotular,
    contra_las_lineas_base,
    error_por_origen,
    guardar,
    medido_contra_predicho,
    rejilla_de_clarke,
)

PARES = [(80 + i % 120, 85 + i % 130) for i in range(300)]
REALES = [r for r, _ in PARES]
PREDICHOS = [p for _, p in PARES]


def _cerrar(figura):
    import matplotlib.pyplot as plt

    plt.close(figura)


# ---------------------------------------------------------------------------
# La rejilla
# ---------------------------------------------------------------------------


def test_01_cada_letra_cae_dentro_de_su_propia_zona():
    """La prueba que importa de este módulo.

    La primera versión ponía la letra en el centro de masa de la zona, y la zona E son dos
    esquinas opuestas: su centro de masa cae en medio del dibujo, **encima de la zona A**. Una
    rejilla que rotula «E» sobre la zona A es peor que una sin rótulos, porque se publica.
    """
    malla = list(range(20, GLUCOSA_MAX + 1))

    rotulos = _donde_rotular(malla)

    assert rotulos
    for letra, x, y in rotulos:
        assert zona(x, y) == letra, f"la «{letra}» cae en la zona {zona(x, y)}, en ({x}, {y})"


def test_02_las_zonas_partidas_se_rotulan_mas_de_una_vez():
    """La E y la D son dos cuadrantes opuestos cada una. Con un solo rótulo, la mitad de la
    figura queda sin decir qué zona es."""
    malla = list(range(20, GLUCOSA_MAX + 1))

    cuantas = {letra: 0 for letra in ZONAS}
    for letra, _, _ in _donde_rotular(malla):
        cuantas[letra] += 1

    assert cuantas["E"] >= 2
    assert cuantas["D"] >= 2
    assert all(cuantas[letra] >= 1 for letra in ZONAS), cuantas


def test_03_el_titulo_lleva_el_ab_su_meta_la_n_y_las_cinco_zonas():
    """Un A+B suelto no se puede leer: ni contra qué meta, ni sobre cuántos pares."""
    figura = rejilla_de_clarke(REALES, PREDICHOS)

    titulo = figura.axes[0].get_title()
    assert "A+B" in titulo and "95" in titulo
    assert f"n = {len(REALES)}" in titulo
    for letra in ZONAS:
        assert f"{letra} " in titulo
    _cerrar(figura)


def test_04_sin_pares_truena_en_lugar_de_dibujar_un_eje_vacio():
    with pytest.raises(ValueError, match="dibujar"):
        rejilla_de_clarke([], [])


def test_05_el_recorte_por_omision_deja_ver_los_puntos():
    """Una rejilla dibujada hasta 600 con datos que no pasan de 200 deja todo apelotonado en
    una esquina, y es la figura que se acaba pegando al documento."""
    figura = rejilla_de_clarke(REALES, PREDICHOS)

    _, tope = figura.axes[0].get_xlim()
    assert max(max(REALES), max(PREDICHOS)) < tope <= 300
    _cerrar(figura)


def test_06_el_recorte_se_puede_forzar_hasta_el_rango_completo():
    figura = rejilla_de_clarke(REALES, PREDICHOS, hasta=GLUCOSA_MAX)

    assert figura.axes[0].get_xlim()[1] == GLUCOSA_MAX
    _cerrar(figura)


# ---------------------------------------------------------------------------
# La serie
# ---------------------------------------------------------------------------


def test_10_lo_medido_sale_continuo_y_lo_predicho_punteado():
    """La regla de interfaz del proyecto, y la que el simulador de la fase 4 rompió sin que una
    revisión de código lo viera: una persona medio dormida tiene que distinguirlas sin leer la
    leyenda."""
    figura = medido_contra_predicho([0, 5, 10], [100, 110, 120], [102, 108, 125])

    lineas = {linea.get_label(): linea for linea in figura.axes[0].lines}
    assert not lineas["Medido"].is_dashed()
    assert lineas["Predicho"].is_dashed()
    _cerrar(figura)


def test_11_los_dos_umbrales_estan_dibujados_y_rotulados():
    figura = medido_contra_predicho([0, 5], [100, 110], [102, 108], umbral_hipo=65)

    textos = [t.get_text() for t in figura.axes[0].texts]
    assert any("65" in t for t in textos)
    assert any("180" in t for t in textos)
    _cerrar(figura)


# ---------------------------------------------------------------------------
# El error por origen
# ---------------------------------------------------------------------------


def test_20_el_peor_origen_queda_arriba():
    """Es la figura del riesgo R-2: si el error es peor para quien no tiene sensor, tiene que
    saltar a la vista, no esconderse en la última fila."""
    figura = error_por_origen(
        {"sensor": 11.0, "manual": 20.0, "fotoGlucometro": 25.0},
        n_por_origen={"sensor": 1000, "manual": 50, "fotoGlucometro": 20},
    )

    etiquetas = [t.get_text() for t in figura.axes[0].get_yticklabels()]
    assert etiquetas[0].startswith("fotoGlucometro")
    assert "n = 20" in etiquetas[0]
    _cerrar(figura)


def test_21_sin_origenes_truena():
    with pytest.raises(ValueError, match="dibujar"):
        error_por_origen({}, n_por_origen={})


# ---------------------------------------------------------------------------
# Las líneas base
# ---------------------------------------------------------------------------


def test_30_el_titulo_nombra_a_los_que_no_le_ganan_a_la_persistencia():
    """«Un modelo que no le gana a la persistencia no justifica su existencia» es una frase del
    protocolo; esta figura la vuelve comprobable de un vistazo."""
    figura = contra_las_lineas_base(
        {"m1": 12.0, "persistencia": 15.0, "ridge": 18.0, "lstm": 14.0},
        el_del_proyecto="m1",
    )

    titulo = figura.axes[0].get_title()
    assert "ridge" in titulo
    assert "lstm" not in titulo
    _cerrar(figura)


def test_31_cuando_todos_le_ganan_el_titulo_lo_dice_en_lugar_de_callarse():
    figura = contra_las_lineas_base(
        {"m1": 12.0, "persistencia": 15.0, "ridge": 14.0},
        el_del_proyecto="m1",
    )

    assert "Todos le ganan" in figura.axes[0].get_title()
    _cerrar(figura)


def test_315_sin_modelo_entrenado_la_figura_no_presume_de_ganarse_a_si_misma():
    """Hoy la única barra es la persistencia. Decir «todos le ganan a la persistencia» con una
    sola barra, que es ella, sería una figura que se felicita sola."""
    figura = contra_las_lineas_base({"persistencia": 15.0}, el_del_proyecto="persistencia")

    titulo = figura.axes[0].get_title()
    assert "Todavía no hay modelo" in titulo
    assert "Todos le ganan" not in titulo
    _cerrar(figura)


def test_32_el_modelo_del_proyecto_se_distingue_sin_color():
    """El proyecto se revisa en escala de grises: el tono no puede ser el único que lo
    distinga."""
    figura = contra_las_lineas_base({"m1": 12.0, "persistencia": 15.0}, el_del_proyecto="m1")

    etiquetas = [t.get_text() for t in figura.axes[0].get_yticklabels()]
    assert "m1 *" in etiquetas
    _cerrar(figura)


def test_33_un_modelo_que_no_esta_en_la_tabla_truena():
    """Equivocarse de nombre dibujaría las líneas base sin el modelo y nadie lo notaría: la
    figura sale igual de bien."""
    with pytest.raises(ValueError, match="m2"):
        contra_las_lineas_base({"m1": 12.0, "persistencia": 15.0}, el_del_proyecto="m2")


# ---------------------------------------------------------------------------
# Guardar
# ---------------------------------------------------------------------------


def test_40_guardar_crea_la_carpeta_y_deja_un_png_de_verdad(tmp_path):
    figura = medido_contra_predicho([0, 5], [100, 110], [102, 108])

    ruta = guardar(figura, tmp_path / "figuras" / "serie.png")

    assert ruta.exists()
    assert ruta.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert ruta.stat().st_size > 5000
    _cerrar(figura)
