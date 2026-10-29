"""La conversión del artefacto a Core ML, y la comprobación de que predice lo mismo.

Necesitan xgboost y coremltools y **se saltan solas** cuando no están, que es como corre la
integración continua. El recorrido del bosque, que no necesita nada, está en
`test_arbol_de_coreml.py`.
"""

import json
import math

import pytest

xgb = pytest.importorskip("xgboost", reason="la conversión necesita xgboost")
ct = pytest.importorskip("coremltools", reason="la conversión necesita coremltools")
np = pytest.importorskip("numpy")

from src.export.arbol import ensamble_de  # noqa: E402
from src.export.to_coreml import (  # noqa: E402
    TOLERANCIA_MG_DL,
    a_32_bits,
    base_de,
    comprobar,
    convertir,
    exportar,
)

# La conversión de verdad -----------------------------------------------------

xgb = pytest.importorskip("xgboost", reason="la conversión necesita xgboost")
ct = pytest.importorskip("coremltools", reason="la conversión necesita coremltools")
np = pytest.importorskip("numpy")


CARACTERISTICAS = ["cob", "iob", "rezago_5", "rezago_30", "hora_sin", "hora_cos"]


def datos(semilla: int = 7, n: int = 600, con_huecos: bool = True):
    rng = np.random.default_rng(semilla)
    escala = np.array([30, 3, 40, 60, 1, 1])
    centro = np.array([30, 2, 120, 120, 0, 0])
    x = rng.normal(size=(n, 6)) * escala + centro
    y = 120 + x[:, 0] * 0.4 - x[:, 1] * 12 + x[:, 2] * 0.3 + rng.normal(scale=4, size=n)
    if con_huecos:
        # Huecos solo en los rezagos, que es donde la serie real los tiene.
        hueco = rng.random((n, 6)) < 0.3
        hueco[:, [0, 1, 4, 5]] = False
        x[hueco] = np.nan
    return x.astype(np.float32), y


def entrenar(x, y, **extra):
    modelo = xgb.XGBRegressor(n_estimators=30, max_depth=4, **extra)
    modelo.fit(x, y)
    return modelo


def test_el_modelo_convertido_predice_lo_mismo_que_xgboost():
    x, y = datos()
    modelo = entrenar(x, y)
    spec = convertir(modelo, CARACTERISTICAS).get_spec()

    comprobacion = comprobar(spec, modelo.predict, x.tolist())

    assert comprobacion.la_traduccion_es_fiel, comprobacion.a_texto()
    assert comprobacion.max_diferencia <= TOLERANCIA_MG_DL


def test_el_valor_de_base_aprendido_de_los_datos_llega_al_modelo_convertido():
    """La prueba de la falla más grave que encontró la comprobación.

    Desde XGBoost 2.0, sin `base_score` fijo el modelo lo aprende de los datos; con glucosa,
    eso es la media de la serie. `coremltools` escribe 0.5 en el spec, que era el valor por
    omisión de XGBoost 1.x. Sin corregirlo, **el modelo del teléfono predice la media por
    debajo del que se midió, siempre**, y nada falla: una glucosa de 140 sale como −3.
    """
    x, y = datos(con_huecos=False)
    modelo = entrenar(x, y)
    aprendida = base_de(modelo)
    assert aprendida > 100, "con glucosa, el valor de base aprendido es del orden de la media"

    sin_corregir = ct.converters.xgboost.convert(
        modelo, feature_names=CARACTERISTICAS, mode="regressor", force_32bit_float=True
    ).get_spec()
    assert list(sin_corregir.treeEnsembleRegressor.treeEnsemble.basePredictionValue) == [0.5]

    corregido = convertir(modelo, CARACTERISTICAS).get_spec()
    assert ensamble_de(corregido).base == pytest.approx(aprendida, abs=1e-3)
    # Y el efecto: sin la corrección, la diferencia es del tamaño del valor de base.
    sin = comprobar(sin_corregir, modelo.predict, x.tolist())
    assert sin.max_diferencia > aprendida - 10


def test_la_marca_del_faltante_viaja_con_la_misma_cuenta_que_el_volcado_de_xgboost():
    """Si la conversión perdiera la dirección de los `NaN`, esta cuenta no cuadraría."""
    import re

    x, y = datos()
    modelo = entrenar(x, y)
    al_verdadero = sum(
        1
        for volcado in modelo.get_booster().get_dump()
        for yes, _no, falta in re.findall(r"yes=(\d+),no=(\d+),missing=(\d+)", volcado)
        if falta == yes
    )

    ensamble = ensamble_de(convertir(modelo, CARACTERISTICAS).get_spec())

    assert ensamble.cortes_con_faltante_al_verdadero == al_verdadero
    assert al_verdadero > 0, "sin huecos en los datos esta prueba no comprobaría nada"


def test_la_comprobacion_avisa_de_que_la_app_tiene_que_mandar_float_de_32_bits():
    """Los umbrales del árbol son de 32 bits y XGBoost compara con esa precisión.

    Mandar `Double` desde Swift voltea algún corte y mueve la predicción varios mg/dL. No es
    una tolerancia que se pueda ignorar: es una instrucción para el código de la app.
    """
    # Un modelo de una sola característica, para que el corte de la raíz se toque siempre, y
    # una fila **construida** para caer justo en el umbral: el valor en doble precisión está
    # un paso por debajo del umbral, y al redondearlo a 32 bits da el umbral exacto. XGBoost
    # redondea y se va por un lado; quien compare en doble precisión se va por el otro.
    # Se construye en lugar de buscarlo al azar porque al azar casi nunca pasa, y «casi
    # nunca» en una prueba significa que un día falla sola y nadie sabe por qué.
    uno = np.linspace(40, 300, 400, dtype=np.float32).reshape(-1, 1)
    escalon = (uno[:, 0] > 150).astype(np.float64) * 60 + 100
    de_una = xgb.XGBRegressor(n_estimators=5, max_depth=1)
    de_una.fit(uno, escalon)
    spec = convertir(de_una, ["rezago_5"]).get_spec()

    umbral = ensamble_de(spec).nodos[(0, 0)].umbral
    justo_abajo = math.nextafter(umbral, -math.inf)
    assert a_32_bits(justo_abajo) == umbral, "el paso de 64 bits tiene que caber en el de 32"

    comprobacion = comprobar(spec, de_una.predict, [[justo_abajo]])

    assert comprobacion.la_traduccion_es_fiel
    assert comprobacion.hay_que_mandar_float_de_32_bits
    assert "Float (32 bits)" in comprobacion.a_texto()


def test_a_32_bits_redondea_igual_que_numpy():
    for valor in (0.1, 1 / 3, 123.456789, -0.0, 1e-20, 1e20):
        assert a_32_bits(valor) == float(np.float32(valor))
    assert math.isnan(a_32_bits(math.nan))


def test_las_caracteristicas_y_el_aviso_medico_viajan_dentro_del_modelo():
    """El orden de las características es el contrato con Swift: una columna corrida da
    números plausibles y equivocados sin que nada falle."""
    x, y = datos()
    convertido = convertir(entrenar(x, y), CARACTERISTICAS)

    metadatos = dict(convertido.get_spec().description.metadata.userDefined)

    assert metadatos["glucy.caracteristicas"].split(",") == CARACTERISTICAS
    assert metadatos["glucy.unidad"] == "mg/dL"
    assert "no es un dispositivo medico certificado" in metadatos["glucy.aviso"]
    assert "dosis de insulina" in convertido.get_spec().description.metadata.shortDescription


def test_comprobar_revienta_sin_filas():
    x, y = datos()
    spec = convertir(entrenar(x, y), CARACTERISTICAS).get_spec()
    with pytest.raises(ValueError, match="no hay filas"):
        comprobar(spec, lambda f: [], [])


def test_ensamble_de_rechaza_lo_que_no_es_un_modelo_de_regresion_por_arboles():
    """Comprobar un clasificador como si fuera regresión daría un número sin sentido."""
    x, _ = datos()
    clasificador = xgb.XGBClassifier(n_estimators=4, max_depth=2)
    clasificador.fit(x, (x[:, 0] > 30).astype(int))
    spec = ct.converters.xgboost.convert(
        clasificador, feature_names=CARACTERISTICAS, mode="classifier", class_labels=[0, 1]
    ).get_spec()
    with pytest.raises(ValueError, match="no es un treeEnsembleRegressor"):
        ensamble_de(spec)


# `exportar`, de punta a punta ------------------------------------------------


def un_artefacto(carpeta, version="v9.9.9"):
    """Entrena, guarda manifiesto y modelo como lo hace `src.train.comun`, y devuelve rutas."""
    from src.eval.metrics import Resultado
    from src.train.artefacto import guardar, manifiesto_de
    from src.train.modelos import XGBoost

    x, y = datos()
    modelo = XGBoost(semilla=1)
    modelo.ajustar(x.tolist(), y.tolist())
    manifiesto = manifiesto_de(
        modelo="m1",
        version=version,
        algoritmo=modelo.algoritmo,
        horizonte_min=30,
        juego="denso",
        caracteristicas=CARACTERISTICAS,
        semilla=1,
        hiperparametros=modelo.hiperparametros,
        origen_datos="sintetico",
        n_entrenamiento=len(x),
        resultado=Resultado(
            n=len(x),
            horizonte_min=30,
            rmse=1.0,
            mae=1.0,
            mard=1.0,
            clarke_ab=99.0,
            clarke_reparto={"A": 99.0},
            sensibilidad_hipo=None,
            falsas_alarmas=0.0,
        ),
        sujetos_entrenamiento=["s1"],
        sujetos_prueba=[],
    )
    guardar(manifiesto, modelo.guardar, carpeta)
    return carpeta / f"m1-{version}.json", x


def test_exportar_escribe_el_modelo_y_su_informe(tmp_path):
    ruta_manifiesto, x = un_artefacto(tmp_path)

    ruta_mlmodel, ruta_informe, comprobacion = exportar(ruta_manifiesto, x.tolist())

    assert ruta_mlmodel.exists() and ruta_informe.exists()
    assert comprobacion.la_traduccion_es_fiel
    informe = json.loads(ruta_informe.read_text())
    assert informe["la_traduccion_es_fiel"] is True
    assert informe["caracteristicas"] == CARACTERISTICAS
    assert informe["comprobacion"]["n"] == len(x)


def test_exportar_no_sobrescribe(tmp_path):
    """Por lo mismo que los artefactos (RF-25): cada predicción dice con qué versión salió."""
    ruta_manifiesto, x = un_artefacto(tmp_path)
    exportar(ruta_manifiesto, x.tolist())

    with pytest.raises(SystemExit, match="no se sobrescribe"):
        exportar(ruta_manifiesto, x.tolist())


def test_exportar_no_escribe_nada_si_la_traduccion_no_es_fiel(tmp_path, monkeypatch):
    """Un `.mlmodel` que no predice lo que se midió es peor que no tenerlo: con el archivo en
    la carpeta, alguien lo mete a la app."""
    ruta_manifiesto, x = un_artefacto(tmp_path)
    import src.export.to_coreml as modulo

    monkeypatch.setattr(modulo, "_con_el_valor_de_base_corregido", lambda c, m: c.get_spec())

    with pytest.raises(SystemExit, match="NO es fiel"):
        exportar(ruta_manifiesto, x.tolist())

    assert not (tmp_path / "m1-v9.9.9.mlmodel").exists()
    assert not (tmp_path / "m1-v9.9.9-coreml.json").exists()


def test_exportar_lo_dice_cuando_falta_el_modelo_junto_al_manifiesto(tmp_path):
    ruta_manifiesto, x = un_artefacto(tmp_path)
    (tmp_path / "m1-v9.9.9.ubj").unlink()

    with pytest.raises(SystemExit, match="falta"):
        exportar(ruta_manifiesto, x.tolist())
