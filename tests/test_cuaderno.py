"""Pruebas de lo que el cuaderno de evaluación calcula antes de dibujar.

Un cuaderno no se prueba, no pasa por el estilo y se rompe sin que nadie se entere hasta que
alguien lo abre, probablemente la semana de la entrega. Por eso lo que decide algo está en
`src/eval/cuaderno.py` y aquí, y el cuaderno solo llama y enseña.

Lo que estas pruebas sostienen es sobre todo **la honestidad de la salida**: que una corrida
sintética no pueda pasar por un resultado.
"""

from pathlib import Path

import pytest

from src.eval.cuaderno import (
    AVISO_SINTETICO,
    HORIZONTE_MIN,
    PASOS_POR_HORIZONTE,
    evaluacion_de_referencia,
    main,
    serie_de_evaluacion,
)

SERIE = [100.0 + (i % 40) for i in range(200)]


def test_01_sin_conjunto_cae_al_generador_de_juguete_y_lo_dice(tmp_path):
    """La caída se devuelve, no se imprime y ya: quien llama tiene que poder marcar la figura,
    y un `print` se pierde entre la salida de una celda."""
    serie, origen = serie_de_evaluacion(tmp_path)

    assert origen == "sintetico"
    assert len(serie) > 100


def test_02_una_corrida_sintetica_no_es_reportable():
    """Es la prueba que importa de este módulo. La diferencia entre una comprobación y un
    resultado del documento final no se puede dejar a la memoria de quien abre el cuaderno."""
    evaluacion = evaluacion_de_referencia(SERIE, "sintetico")

    assert not evaluacion.es_reportable
    assert evaluacion.subtitulo == AVISO_SINTETICO


def test_03_con_ohiot1dm_si_lo_es():
    evaluacion = evaluacion_de_referencia(SERIE, "ohiot1dm")

    assert evaluacion.es_reportable
    assert evaluacion.subtitulo == "ohiot1dm"


def test_10_la_referencia_es_la_persistencia_y_no_otra_cosa():
    """Si esto dejara de ser la persistencia, el listón contra el que se compara M-1 sería
    otro y la comparación del reporte no significaría nada."""
    evaluacion = evaluacion_de_referencia(SERIE, "sintetico")

    assert evaluacion.predichos == SERIE[:-PASOS_POR_HORIZONTE]
    assert evaluacion.reales == SERIE[PASOS_POR_HORIZONTE:]
    assert list(evaluacion.rmse_por_modelo) == ["persistencia"]
    assert evaluacion.resultado.horizonte_min == HORIZONTE_MIN


def test_11_los_minutos_van_de_cinco_en_cinco_y_son_tantos_como_los_pares():
    evaluacion = evaluacion_de_referencia(SERIE, "sintetico")

    assert len(evaluacion.minutos) == len(evaluacion.reales)
    assert evaluacion.minutos[:3] == [0, 5, 10]


def test_12_una_serie_mas_corta_que_el_horizonte_truena_con_su_motivo():
    """Devolver una evaluación vacía dibujaría cuatro figuras en blanco que se ven igual de
    bien que las buenas."""
    with pytest.raises(ValueError, match="no alcanza"):
        evaluacion_de_referencia(SERIE[:PASOS_POR_HORIZONTE], "sintetico")


def test_20_la_orden_deja_las_cuatro_figuras_y_avisa(monkeypatch, capsys, tmp_path):
    pytest.importorskip("matplotlib")
    salida = tmp_path / "figuras"
    monkeypatch.setattr(
        "sys.argv",
        ["cuaderno", "--carpeta", str(tmp_path / "sin-datos"), "--salida", str(salida)],
    )

    main()

    impreso = capsys.readouterr().out
    assert "R-8" in impreso
    assert "A+B" in impreso
    archivos = sorted(p.name for p in Path(salida).glob("*.png"))
    assert archivos == ["clarke.png", "error-por-origen.png", "lineas-base.png", "serie.png"]


def test_30_el_cuaderno_corre_entero(monkeypatch, tmp_path):
    """Ejecuta las celdas de código del cuaderno, en orden y en un mismo espacio de nombres.

    Es la prueba que lo mantiene vivo. Un cuaderno es el único archivo del repositorio que
    nadie compila y nadie prueba: se rompe cuando cambia una firma, y se descubre al abrirlo,
    que con suerte es la semana de la entrega. Aquí se cae la integración continua en su lugar.

    No comprueba qué se dibuja —para eso están las pruebas de `dibujos`—, solo que corre.
    """
    pytest.importorskip("matplotlib")
    import json

    celdas = json.loads(Path("notebooks/evaluacion.ipynb").read_text(encoding="utf-8"))["cells"]
    codigo = [c for c in celdas if c["cell_type"] == "code"]
    assert len(codigo) >= 5, "el cuaderno se quedó sin celdas de código"

    monkeypatch.chdir(tmp_path)  # sin `data/`, así que cae al generador de juguete
    espacio: dict = {}
    for numero, celda in enumerate(codigo, start=1):
        try:
            exec("".join(celda["source"]), espacio)  # noqa: S102
        except Exception as error:  # pragma: no cover - el mensaje es el valor de la prueba
            raise AssertionError(
                f"la celda de código {numero} del cuaderno falló: {error}"
            ) from error

    assert (tmp_path / "artifacts" / "figuras" / "clarke.png").exists()
