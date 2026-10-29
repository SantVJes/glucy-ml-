"""Pruebas del exportador de la serie para HealthKit (plan B de la fase 5).

Lo que importa aquí no es el formato del archivo: es que el archivo contenga **los casos que
rompen las cosas**. Una serie perfecta cada cinco minutos validaría el único escenario que un
sensor comercial nunca produce, y la prueba de 48 horas saldría verde sin haber probado nada.

Todo es biblioteca estándar, así que corre en la integración continua sin instalar nada.
"""

import argparse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from src.datos.para_healthkit import (
    CABECERA,
    HUECOS_LARGOS_MIN,
    RETRASO_MIN,
    ZONA,
    Punto,
    _lecturas_del_origen,
    con_huecos_y_retrasos,
    escribir,
    leer,
    main,
    resumen,
)
from src.datos.sintetico import conjunto
from src.ingest.ohiot1dm import Lectura, Registro

PASO = timedelta(minutes=5)


def _serie(cuantas: int = 600, desde: datetime | None = None) -> list[tuple[datetime, float]]:
    """Una serie regular de cinco minutos, que es lo que el módulo tiene que ensuciar."""
    inicio = desde or datetime(2026, 11, 1, 0, 0)
    return [(inicio + i * PASO, 100.0 + i % 7) for i in range(cuantas)]


# ---------------------------------------------------------------------------
# Los huecos
# ---------------------------------------------------------------------------


def test_01_la_serie_sale_con_huecos_de_los_dos_lados_del_umbral():
    """Quince minutos es el umbral de interpolación (RF-23), y el archivo tiene que ejercitar
    las dos ramas: huecos que la app puede rellenar y huecos que no."""
    puntos = list(con_huecos_y_retrasos(_serie(2000), semilla=3))
    tiempos = [p.ts_medicion for p in puntos]
    huecos = [
        (b - a).total_seconds() / 60
        for a, b in zip(tiempos, tiempos[1:], strict=False)
        if b - a > PASO
    ]

    assert any(h <= 15 for h in huecos), "ningún hueco corto: la rama interpolable no se prueba"
    assert any(h > 15 for h in huecos), "ningún hueco largo: la rama que degrada no se prueba"
    assert max(huecos) <= max(HUECOS_LARGOS_MIN)


def test_02_un_hueco_es_una_lectura_que_falta_y_no_un_valor_nulo():
    """Así es como falta un dato en HealthKit: no hay muestra. Un cero escrito en la base sería
    otra cosa y la app tendría que distinguirlo."""
    puntos = list(con_huecos_y_retrasos(_serie(2000)))

    assert len(puntos) < 2000, "no se quitó ninguna lectura"
    assert all(20 <= p.mg_dl <= 600 for p in puntos)


def test_03_sin_huecos_la_serie_queda_regular():
    """La bandera existe para depurar: si la prueba del teléfono falla, primero se descarta que
    el problema sean los huecos."""
    puntos = list(con_huecos_y_retrasos(_serie(200), con_huecos=False, con_retrasos=False))
    tiempos = [p.ts_medicion for p in puntos]

    assert len(puntos) == 200
    assert {b - a for a, b in zip(tiempos, tiempos[1:], strict=False)} == {PASO}


# ---------------------------------------------------------------------------
# Los retrasos
# ---------------------------------------------------------------------------


def test_10_las_lecturas_atrasadas_llegan_en_bloque_con_una_sola_marca():
    """Es la prueba del módulo.

    Un sensor que vuelve a alcance no escribe sus lecturas una por una y a su hora: las escribe
    **todas de golpe** en el momento de volver (RF-06b). Por eso todas las del mismo bloque
    comparten `ts_escritura`, y por eso cada bloque tiene más de una lectura: una sola no
    probaría que la app aguanta un lote.
    """
    tarde = [p for p in con_huecos_y_retrasos(_serie(1000), semilla=3) if p.llega_tarde]
    bloques: dict[datetime, list[Punto]] = {}
    for punto in tarde:
        bloques.setdefault(punto.ts_escritura, []).append(punto)

    assert bloques, "ninguna lectura llegó tarde"
    for escritura, del_bloque in bloques.items():
        assert len(del_bloque) > 1, escritura
        assert all(p.ts_escritura == escritura for p in del_bloque)
        assert max(p.ts_medicion for p in del_bloque) < escritura


def test_11_el_retraso_no_pasa_de_lo_declarado():
    for punto in con_huecos_y_retrasos(_serie(1000), semilla=3):
        atraso = (punto.ts_escritura - punto.ts_medicion).total_seconds() / 60
        assert 0 <= atraso <= RETRASO_MIN


def test_12_las_escrituras_nunca_van_hacia_atras():
    """La utilidad del iPhone ordena por `ts_escritura` y duerme entre una y otra. Si el archivo
    pidiera escribir en el pasado, esa pausa saldría negativa y la utilidad tendría que decidir
    qué hacer; el archivo no le pasa esa decisión."""
    escrituras = [p.ts_escritura for p in con_huecos_y_retrasos(_serie(2000), semilla=3)]

    assert escrituras == sorted(escrituras)


def test_13_sin_retrasos_todo_se_escribe_a_su_hora():
    puntos = list(con_huecos_y_retrasos(_serie(1000), con_retrasos=False))

    assert puntos, "la serie salió vacía"
    assert not any(p.llega_tarde for p in puntos)


# ---------------------------------------------------------------------------
# Reproducibilidad y zona
# ---------------------------------------------------------------------------


def test_20_la_misma_semilla_da_exactamente_el_mismo_archivo():
    """Una prueba de 48 horas que no se puede repetir no sirve para comparar un arreglo contra
    el problema que arregla."""
    a = list(con_huecos_y_retrasos(_serie(), semilla=9))
    b = list(con_huecos_y_retrasos(_serie(), semilla=9))
    c = list(con_huecos_y_retrasos(_serie(), semilla=10))

    assert a == b
    assert a != c


def test_21_las_marcas_salen_con_zona_aunque_la_serie_de_origen_no_la_traiga():
    """Regla 5 del proyecto. Las dos series de origen vienen sin zona: OhioT1DM es hora local
    del sujeto sin decir cuál, y el generador de juguete tampoco la pone."""
    puntos = list(con_huecos_y_retrasos(_serie(10)))

    assert all(p.ts_medicion.tzinfo is not None for p in puntos)
    assert all(p.ts_escritura.tzinfo is not None for p in puntos)
    assert puntos[0].ts_medicion.tzinfo is ZONA
    assert puntos[0].ts_medicion.utcoffset() == timedelta(hours=-6)


def test_22_una_serie_que_ya_trae_zona_se_respeta():
    madrid = ZoneInfo("Europe/Madrid")
    con_zona = [(datetime(2026, 11, 1, 12, 0, tzinfo=madrid), 110.0)]
    punto = next(iter(con_huecos_y_retrasos(con_zona)))

    assert punto.ts_medicion.tzinfo is madrid
    assert punto.ts_medicion.utcoffset() == timedelta(hours=1)


# ---------------------------------------------------------------------------
# El archivo
# ---------------------------------------------------------------------------


def test_30_el_viaje_redondo_no_pierde_nada(tmp_path):
    """La utilidad del iPhone va a leer esto con un lector de CSV cualquiera, no con este
    módulo: si el viaje redondo pierde la zona o decimales, lo pierde allá también."""
    puntos = list(con_huecos_y_retrasos(_serie(400), semilla=3))
    ruta = tmp_path / "serie.csv"

    assert escribir(ruta, puntos) == len(puntos)
    assert leer(ruta) == puntos


def test_31_la_cabecera_va_en_el_archivo(tmp_path):
    ruta = tmp_path / "serie.csv"
    escribir(ruta, con_huecos_y_retrasos(_serie(10)))

    assert ruta.read_text(encoding="utf-8").splitlines()[0] == ",".join(CABECERA)


def test_32_un_archivo_con_otra_cabecera_no_se_lee_a_medias(tmp_path):
    """Leer un archivo de otro formato columna por columna daría fechas absurdas sin fallar.
    Mejor que truene aquí."""
    ruta = tmp_path / "otra-cosa.csv"
    ruta.write_text("fecha,valor\n2026-11-01T00:00:00,110\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cabecera"):
        leer(ruta)


def test_33_escribir_devuelve_las_filas_aunque_le_pasen_un_generador(tmp_path):
    """Contar un generador después de agotarlo da cero, y quien llama no podría comprobar que
    escribió lo que creía."""
    ruta = tmp_path / "serie.csv"
    filas = escribir(ruta, con_huecos_y_retrasos(_serie(100), con_huecos=False))

    assert filas == 100
    assert len(ruta.read_text(encoding="utf-8").splitlines()) == 101


# ---------------------------------------------------------------------------
# El resumen, que es lo que se copia al reporte
# ---------------------------------------------------------------------------


def test_40_el_resumen_cuenta_lo_que_hay_en_el_archivo():
    puntos = list(con_huecos_y_retrasos(_serie(2000), semilla=3))
    texto = resumen(puntos)

    assert f"{len(puntos)} lecturas" in texto
    assert "huecos" in texto and "más de 15 min" in texto
    assert f"{sum(1 for p in puntos if p.llega_tarde)} lecturas que llegan tarde" in texto


def test_41_un_archivo_vacio_lo_dice_en_lugar_de_reventar():
    assert resumen([]) == "el archivo quedó vacío"


# ---------------------------------------------------------------------------
# De punta a punta con el generador de juguete
# ---------------------------------------------------------------------------


def test_50_dos_dias_de_la_serie_de_juguete_salen_completos(tmp_path):
    """Dos días es lo que pide la fase 5. Se comprueba el largo porque un exportador que
    recorta a un día dejaría la prueba del teléfono a medias sin avisar."""
    registro = next(iter(conjunto(1, dias=3, semilla=5)))
    lecturas = sorted((lec.ts, lec.mg_dl) for lec in registro.lecturas if lec.del_sensor)
    hasta = lecturas[0][0] + timedelta(days=2)
    puntos = list(con_huecos_y_retrasos([(ts, mg) for ts, mg in lecturas if ts <= hasta]))

    horas = (puntos[-1].ts_medicion - puntos[0].ts_medicion).total_seconds() / 3600
    assert 47 <= horas <= 48
    assert escribir(tmp_path / "serie.csv", puntos) == len(puntos)


# ---------------------------------------------------------------------------
# Qué lecturas entran, y la línea de órdenes
# ---------------------------------------------------------------------------


def test_60_los_pinchazos_sueltos_no_entran_al_archivo(monkeypatch):
    """Solo la serie del sensor continuo.

    Escribir tres pinchazos en HealthKit como si fueran una serie le haría creer a la app que
    tiene sensor cuando no lo tiene, y el modo sensor es justo lo que esta prueba valida.
    """
    inicio = datetime(2026, 11, 1, 0, 0)
    mezcla = Registro(
        sujeto="559",
        lecturas=tuple(
            Lectura(sujeto="559", ts=inicio + i * PASO, mg_dl=100.0 + i, del_sensor=i % 2 == 0)
            for i in range(20)
        ),
        eventos={},
    )
    monkeypatch.setattr("src.datos.sintetico.conjunto", lambda *a, **k: iter([mezcla]))

    lecturas = _lecturas_del_origen(
        argparse.Namespace(datos="sintetico", dias=1, semilla=5, carpeta="data", sujeto="")
    )

    assert [mg for _, mg in lecturas] == [100.0 + i for i in range(0, 20, 2)]


def test_61_la_linea_de_ordenes_escribe_el_archivo_y_avisa_del_riesgo(
    monkeypatch, capsys, tmp_path
):
    """El aviso no es decorativo: sin él, 48 horas de serie sintética se cuentan en el reporte
    como si hubieran validado un sensor comercial (R-8)."""
    salida = tmp_path / "serie.csv"
    monkeypatch.setattr(
        "sys.argv",
        ["para_healthkit", "--dias", "1", "--salida", str(salida)],
    )

    main()

    impreso = capsys.readouterr().out
    assert "lecturas en 24" in impreso
    assert "R-8" in impreso
    assert len(leer(salida)) > 200
