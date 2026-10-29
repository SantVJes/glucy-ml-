"""Pruebas del artefacto y su manifiesto.

Todas corren en la integración continua porque el módulo es de biblioteca estándar: la lógica
del manifiesto es justo lo que conviene tener probado, ya que es lo que hace repetible una
corrida y lo que el backend lee antes de cargar un modelo.
"""

import json

import pytest

from src.eval.metrics import evaluar
from src.train.artefacto import (
    Manifiesto,
    guardar,
    leer_manifiesto,
    manifiesto_de,
    ruta_del_artefacto,
)

REALES = [100, 60, 150, 200]
PREDICHOS = [105, 65, 160, 190]


def manifiesto(**sobre):
    base = {
        "modelo": "m1",
        "version": "v0.1.0",
        "algoritmo": "xgboost",
        "horizonte_min": 30,
        "juego": "denso",
        "caracteristicas": ["mg_dl", "rezago_5", "cob"],
        "semilla": 42,
        "hiperparametros": {"max_depth": 4},
        "origen_datos": "ohiot1dm",
        "n_entrenamiento": 1000,
        "resultado": evaluar(REALES, PREDICHOS, horizonte_min=30),
        "sujetos_entrenamiento": ["S00", "S01"],
        "sujetos_prueba": ["S02"],
    }
    return manifiesto_de(**{**base, **sobre})


# --- Lo que el manifiesto no deja pasar -------------------------------------------------------


def test_un_manifiesto_sin_caracteristicas_revienta():
    """Es la lista que `GET /predict` compara antes de cargar el artefacto.

    Sin ella, una columna desplazada da números plausibles y equivocados y nada falla.
    """
    with pytest.raises(ValueError, match="lista de características"):
        manifiesto(caracteristicas=[])


def test_un_sujeto_en_los_dos_lados_revienta():
    """El manifiesto no debe poder afirmar algo que invalidaría la métrica."""
    with pytest.raises(ValueError, match="están en los dos lados"):
        manifiesto(sujetos_entrenamiento=["S00", "S01"], sujetos_prueba=["S01"])


def test_un_origen_de_datos_desconocido_revienta():
    with pytest.raises(ValueError, match="origen de datos desconocido"):
        manifiesto(origen_datos="kaggle")


# --- Lo que se puede reportar y lo que no ----------------------------------------------------


def test_solo_las_metricas_de_datos_reales_se_pueden_reportar():
    """Un artefacto entrenado con las series de juguete queda marcado, y el marcaje importa.

    En esas series la señal la generó una suma de senos: cualquier modelo decente sale
    excelente, y ese número en el documento final sería un resultado inventado sin que nadie
    mintiera a propósito.
    """
    assert manifiesto(origen_datos="ohiot1dm").se_puede_reportar
    assert not manifiesto(origen_datos="sintetico").se_puede_reportar
    assert not manifiesto(origen_datos="simulador").se_puede_reportar


# --- Las rutas, donde ya hubo un error -------------------------------------------------------


def test_el_nombre_del_artefacto_conserva_la_version_completa():
    """`v0.1.0` tiene que quedar entero en el nombre.

    `Path.with_suffix(".json")` trataba el `.0` del parche como extensión y lo reemplazaba, así
    que el manifiesto salía como `m1-v0.1.json`: con un nombre distinto del `.ubj` del modelo y
    pisando el de cualquier otra versión de parche de la misma menor.
    """
    base = ruta_del_artefacto("artifacts", "m1", "v0.1.0")
    assert base.name == "m1-v0.1.0"


def test_guardar_escribe_el_manifiesto_con_su_version_entera(tmp_path):
    rutas = guardar(manifiesto(), lambda base: None, tmp_path)
    assert rutas["manifiesto"].name == "m1-v0.1.0.json"
    assert rutas["manifiesto"].exists()


def test_guardar_tambien_devuelve_la_ruta_del_modelo(tmp_path):
    def guardar_modelo(base):
        ruta = f"{base}.ubj"
        open(ruta, "wb").close()
        return ruta

    rutas = guardar(manifiesto(), guardar_modelo, tmp_path)
    assert rutas["modelo"].name == "m1-v0.1.0.ubj"
    assert rutas["manifiesto"].stem == rutas["modelo"].stem


def test_no_se_sobrescribe_un_artefacto_que_ya_existe(tmp_path):
    """Cada predicción guarda con qué versión se calculó (RF-25); pisar una rompe ese rastro."""
    guardar(manifiesto(), lambda base: None, tmp_path)
    with pytest.raises(FileExistsError, match="no se sobrescriben"):
        guardar(manifiesto(), lambda base: None, tmp_path)


def test_dos_versiones_distintas_conviven(tmp_path):
    guardar(manifiesto(version="v0.1.0"), lambda base: None, tmp_path)
    guardar(manifiesto(version="v0.1.1"), lambda base: None, tmp_path)
    nombres = sorted(p.name for p in tmp_path.iterdir())
    assert nombres == ["m1-v0.1.0.json", "m1-v0.1.1.json"]


# --- Ir y volver -----------------------------------------------------------------------------


def test_el_manifiesto_va_y_vuelve_igual(tmp_path):
    original = manifiesto()
    rutas = guardar(original, lambda base: None, tmp_path)
    assert leer_manifiesto(rutas["manifiesto"]) == original


def test_leer_un_manifiesto_editado_a_mano_aplica_las_comprobaciones(tmp_path):
    """Un archivo con un sujeto en los dos lados falla al leerse, no al usarse."""
    rutas = guardar(manifiesto(), lambda base: None, tmp_path)
    datos = json.loads(rutas["manifiesto"].read_text(encoding="utf-8"))
    datos["sujetos_prueba"] = ["S00"]
    rutas["manifiesto"].write_text(json.dumps(datos), encoding="utf-8")
    with pytest.raises(ValueError, match="están en los dos lados"):
        leer_manifiesto(rutas["manifiesto"])


def test_un_manifiesto_de_una_version_mas_nueva_lo_dice(tmp_path):
    """Mejor un error que decirlo que ignorar un campo que no se entiende."""
    ruta = tmp_path / "raro.json"
    ruta.write_text(json.dumps({"modelo": "m1", "campo_del_futuro": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="campos que este código no conoce"):
        leer_manifiesto(ruta)


# --- Lo que lleva dentro ----------------------------------------------------------------------


def test_el_manifiesto_guarda_las_metricas_de_las_lineas_base_al_lado():
    """«RMSE 18» no significa nada sin el «la persistencia dio 22» al lado."""
    persistencia = evaluar([100, 60, 150, 200], [100, 60, 150, 200], horizonte_min=30)
    m = manifiesto(lineas_base={"persistencia": persistencia})
    assert m.metricas_lineas_base["persistencia"]["rmse"] == 0.0


def test_el_manifiesto_guarda_las_metricas_de_cada_sujeto():
    """Con doce personas, el promedio esconde a aquella para la que el sistema funciona peor."""
    uno = evaluar(REALES, PREDICHOS, horizonte_min=30)
    m = manifiesto(por_sujeto={"S02": uno})
    assert m.metricas_por_sujeto["S02"]["n"] == len(REALES)


def test_el_protocolo_explica_por_que_no_hay_sujetos_de_prueba():
    """Sin esa línea, una lista vacía de sujetos de prueba se lee como «no se evaluó»."""
    m = manifiesto(sujetos_prueba=[], protocolo="LOSO sobre 12 sujetos; el artefacto usa todos")
    assert "LOSO" in m.protocolo


def test_el_manifiesto_anota_el_entorno():
    """Una diferencia de versión de la biblioteca cambia los números y es lo primero a descartar."""
    m = manifiesto()
    assert "python" in m.entorno
    assert "sistema" in m.entorno


def test_el_json_es_legible_por_una_persona():
    texto = manifiesto().a_json()
    assert texto.startswith("{\n")
    assert '"semilla": 42' in texto
    assert json.loads(texto)["version"] == "v0.1.0"


def test_el_manifiesto_es_inmutable():
    """Para que nadie le cambie la lista de características después de escribirlo."""
    m = manifiesto()
    with pytest.raises(AttributeError):
        m.semilla = 7  # type: ignore[misc]
    assert isinstance(m, Manifiesto)
