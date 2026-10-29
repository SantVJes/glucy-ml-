"""Pruebas de los modelos y las líneas base.

Las dos líneas base que no se ajustan están escritas con biblioteca estándar, así que **se
prueban en la integración continua**. Las tres que necesitan xgboost o scikit-learn se saltan
cuando no están instaladas, porque la CI instala solo pytest y ruff a propósito: si las pruebas
de las reglas clínicas dependieran de una rueda de 200 MB, dejarían de correr en cada pull
request, que es justo cuando valen.
"""

import math

import pytest

from src.features.features import CARACTERISTICAS_DENSAS
from src.train.modelos import ExtrapolacionLineal, Persistencia

I_VALOR = CARACTERISTICAS_DENSAS.index("mg_dl")
I_REZAGO_30 = CARACTERISTICAS_DENSAS.index("rezago_30")
I_DERIVADA = CARACTERISTICAS_DENSAS.index("derivada_1")


def fila(**valores: float) -> list[float]:
    """Una fila densa con `NaN` en todo menos en lo que se pida, como las de verdad."""
    v = [math.nan] * len(CARACTERISTICAS_DENSAS)
    for nombre, valor in valores.items():
        v[CARACTERISTICAS_DENSAS.index(nombre)] = valor
    return v


# --- Persistencia ---------------------------------------------------------------------------


def test_la_persistencia_devuelve_el_valor_de_ahora():
    modelo = Persistencia()
    x = [fila(mg_dl=120.0), fila(mg_dl=85.0)]
    assert modelo.predecir(x) == [120.0, 85.0]


def test_la_persistencia_no_mira_ninguna_otra_columna():
    """Es lo que la hace una línea base y no un modelo."""
    modelo = Persistencia()
    sin_nada = fila(mg_dl=150.0)
    con_todo = fila(mg_dl=150.0, rezago_5=10.0, cob=99.0, iob=9.0, derivada_1=5.0)
    assert modelo.predecir([sin_nada]) == modelo.predecir([con_todo])


def test_ajustar_la_persistencia_no_hace_nada_y_no_revienta():
    """Entra al mismo bucle que todo lo demás, así que tiene que aceptar que la ajusten."""
    modelo = Persistencia()
    modelo.ajustar([fila(mg_dl=100.0)], [110.0])
    assert modelo.predecir([fila(mg_dl=100.0)]) == [100.0]


# --- Extrapolación lineal --------------------------------------------------------------------


def test_la_extrapolacion_proyecta_la_pendiente_de_la_ventana():
    """A 30 minutos con pendiente de 30: de 150 con 120 hace media hora, proyecta 180."""
    modelo = ExtrapolacionLineal(30)
    x = [fila(mg_dl=150.0, rezago_30=120.0)]
    assert modelo.predecir(x) == pytest.approx([180.0])


def test_la_ventana_de_la_pendiente_es_el_horizonte_por_omision():
    """Es la versión defendible: «el ritmo que lleva» medido en el mismo plazo que se proyecta.

    Y es la decisión que más mueve a esta línea base: con la pendiente de cinco minutos su RMSE
    se triplica, y **una línea base pésima adula al modelo**.
    """
    assert ExtrapolacionLineal(30).hiperparametros["pendiente_min"] == 30
    assert ExtrapolacionLineal(60).hiperparametros["pendiente_min"] == 60


def test_la_ventana_de_la_pendiente_se_puede_cambiar():
    modelo = ExtrapolacionLineal(30, pendiente_min=15)
    # De 150, con 135 hace quince minutos: 1 mg/dL por minuto, así que a 30 min son 180.
    x = [fila(mg_dl=150.0, rezago_15=135.0)]
    assert modelo.predecir(x) == pytest.approx([180.0])
    assert modelo.hiperparametros["pendiente_min"] == 15


def test_una_ventana_que_no_existe_revienta_y_dice_cuales_hay():
    with pytest.raises(ValueError, match="las ventanas disponibles son"):
        ExtrapolacionLineal(30, pendiente_min=45)


def test_sin_el_rezago_devuelve_nan_y_no_cae_a_la_persistencia():
    """Rellenar la convertiría en silencio en la otra línea base, y la haría ver mejor."""
    modelo = ExtrapolacionLineal(30)
    [salida] = modelo.predecir([fila(mg_dl=150.0)])  # rezago_30 es NaN
    assert math.isnan(salida)


def test_una_serie_plana_extrapola_a_lo_mismo():
    modelo = ExtrapolacionLineal(30)
    assert modelo.predecir([fila(mg_dl=120.0, rezago_30=120.0)]) == pytest.approx([120.0])


def test_una_bajada_se_proyecta_hacia_abajo():
    """El caso que importa: una bajada que sigue es la que lleva a una hipoglucemia."""
    modelo = ExtrapolacionLineal(30)
    [salida] = modelo.predecir([fila(mg_dl=100.0, rezago_30=160.0)])
    assert salida == pytest.approx(40.0)


# --- Los tres que se ajustan -----------------------------------------------------------------


def _datos_de_juguete(n: int = 200):
    """Filas con **todas** las columnas pobladas y `NaN` solo en algunas filas.

    Así son las de verdad, y la diferencia importa: con una columna entera en `NaN`,
    `HistGradientBoostingRegressor` revienta con «window shape cannot be larger than input
    array shape» y el imputador de Ridge avisa de que no tiene con qué imputar. Eran mis datos
    de prueba los irreales, no los modelos; una columna que nunca trae valor no debería estar
    en el juego de características.

    No es un conjunto de evaluación: lo único que se comprueba es que cada modelo ajuste,
    prediga el número correcto de valores y no devuelva `NaN`. Si aprende bien o mal se mide
    con los datos de verdad, no aquí.
    """
    x, y = [], []
    for i in range(n):
        valor = 80.0 + (i % 100)
        pendiente = float((i % 7) - 3)
        campos = {
            "mg_dl": valor,
            "rezago_5": valor - pendiente,
            "rezago_10": valor - pendiente * 2,
            "rezago_15": valor - pendiente * 3,
            "rezago_30": valor - pendiente * 6,
            "rezago_60": valor - pendiente * 12,
            "derivada_1": pendiente / 5,
            "derivada_2": 0.0,
            "media_30": valor - pendiente * 3,
            "media_60": valor - pendiente * 6,
            "cob": float(i % 40),
            "iob": float(i % 5),
            "hora_sin": math.sin(i / 10),
            "hora_cos": math.cos(i / 10),
            "dia_semana": float(i % 7),
        }
        # Una fila de cada diez pierde su rezago más largo, como al empezar una serie.
        if i % 10 == 0:
            campos["rezago_60"] = math.nan
            campos["media_60"] = math.nan
        x.append(fila(**campos))
        y.append(valor + pendiente * 6)
    return x, y


@pytest.mark.parametrize("nombre", ["XGBoost", "Ridge", "Bosque"])
def test_los_modelos_que_se_ajustan_predicen_un_numero_por_fila(nombre):
    pytest.importorskip("numpy")
    if nombre == "XGBoost":
        pytest.importorskip("xgboost")
        from src.train.modelos import XGBoost as Clase
    elif nombre == "Ridge":
        pytest.importorskip("sklearn")
        from src.train.modelos import Ridge as Clase
    else:
        pytest.importorskip("sklearn")
        from src.train.modelos import Bosque as Clase

    x, y = _datos_de_juguete()
    modelo = Clase(semilla=42)
    modelo.ajustar(x, y)
    salida = modelo.predecir(x[:20])
    assert len(salida) == 20
    assert not any(math.isnan(v) for v in salida)


@pytest.mark.parametrize("nombre", ["XGBoost", "Ridge", "Bosque"])
def test_predecir_sin_ajustar_revienta(nombre):
    """Mejor un error que una predicción de un modelo vacío."""
    if nombre == "XGBoost":
        from src.train.modelos import XGBoost as Clase
    elif nombre == "Ridge":
        from src.train.modelos import Ridge as Clase
    else:
        from src.train.modelos import Bosque as Clase

    with pytest.raises(RuntimeError, match="ajustar"):
        Clase(semilla=42).predecir([fila(mg_dl=100.0)])


def test_la_misma_semilla_da_el_mismo_modelo():
    """Un resultado que no se puede repetir no sirve para un trabajo que se va a evaluar."""
    pytest.importorskip("xgboost")
    from src.train.modelos import XGBoost

    x, y = _datos_de_juguete()
    primero, segundo = XGBoost(semilla=7), XGBoost(semilla=7)
    primero.ajustar(x, y)
    segundo.ajustar(x, y)
    assert primero.predecir(x[:50]) == segundo.predecir(x[:50])


def test_la_semilla_queda_en_los_hiperparametros():
    """Es lo que va al manifiesto, y sin ella la corrida no se puede repetir."""
    from src.train.modelos import XGBoost

    assert XGBoost(semilla=123).hiperparametros["random_state"] == 123


def test_los_hiperparametros_se_pueden_sobreescribir():
    from src.train.modelos import XGBoost

    modelo = XGBoost(semilla=1, max_depth=9)
    assert modelo.hiperparametros["max_depth"] == 9
    assert modelo.hiperparametros["n_estimators"] == XGBoost.POR_OMISION["n_estimators"]


def test_xgboost_recibe_los_nan_tal_cual():
    """Aprende para qué lado mandarlos; convertirlos a cero sería perder información."""
    pytest.importorskip("xgboost")
    from src.train.modelos import XGBoost

    x, y = _datos_de_juguete()
    modelo = XGBoost(semilla=42)
    modelo.ajustar(x, y)
    # La fila trae NaN en la mayoría de las columnas y aun así predice.
    [salida] = modelo.predecir([fila(mg_dl=120.0)])
    assert not math.isnan(salida)


def test_guardar_xgboost_deja_un_archivo_con_la_version_en_el_nombre(tmp_path):
    pytest.importorskip("xgboost")
    from src.train.modelos import XGBoost

    x, y = _datos_de_juguete()
    modelo = XGBoost(semilla=42)
    modelo.ajustar(x, y)
    ruta = modelo.guardar(tmp_path / "m1-v0.1.0")
    assert ruta.endswith("m1-v0.1.0.ubj")


# --- La quinta línea base, la recurrente ----------------------------------------------------


def test_la_lstm_arma_su_secuencia_con_los_rezagos_de_la_fila():
    """Sin torch tampoco: la forma de la secuencia es aritmética, no red.

    Seis pasos —`rezago_60, rezago_30, rezago_15, rezago_10, rezago_5, mg_dl`— y en cada paso
    la glucosa de ese instante más el contexto repetido. Es lo que hace que la red vea la misma
    información que XGBoost sobre las mismas filas.
    """
    pytest.importorskip("numpy")
    import numpy as np

    from src.train.modelos import LSTM

    modelo = LSTM(semilla=1)
    # Se ajustan a mano el centro y la escala para mirar solo el armado.
    modelo._centro = np.zeros(len(CARACTERISTICAS_DENSAS))
    modelo._escala = np.ones(len(CARACTERISTICAS_DENSAS))

    fila = [0.0] * len(CARACTERISTICAS_DENSAS)
    for nombre, valor in (
        ("rezago_60", 60.0),
        ("rezago_30", 30.0),
        ("rezago_15", 15.0),
        ("rezago_10", 10.0),
        ("rezago_5", 5.0),
        ("mg_dl", 1.0),
        ("cob", 42.0),
    ):
        fila[CARACTERISTICAS_DENSAS.index(nombre)] = valor

    secuencias = modelo._secuencias([fila])

    assert secuencias.shape == (1, 6, len(CARACTERISTICAS_DENSAS) - 5)
    # La primera columna de cada paso es la glucosa, del punto más viejo al más nuevo.
    assert list(secuencias[0, :, 0]) == [60.0, 30.0, 15.0, 10.0, 5.0, 1.0]
    # Y el contexto se repite igual en los seis pasos.
    assert len({tuple(paso[1:]) for paso in secuencias[0]}) == 1
    assert 42.0 in secuencias[0, 0, 1:]


def test_la_lstm_se_niega_con_el_juego_disperso():
    """Sobre un solo punto no hay serie que recorrer, y una LSTM ahí no es una LSTM."""
    from src.features.features import CARACTERISTICAS_DISPERSAS
    from src.train.modelos import LSTM

    with pytest.raises(ValueError, match="no hay serie"):
        LSTM(semilla=1, caracteristicas=CARACTERISTICAS_DISPERSAS)


def test_la_lstm_imputa_los_faltantes_con_la_mediana_del_entrenamiento():
    """Una red no acepta `NaN`, así que la imputación es parte de esta línea base y conviene
    recordarlo al leer su número: si sale mal, parte puede ser la mediana."""
    pytest.importorskip("numpy")
    import numpy as np

    from src.train.modelos import LSTM

    modelo = LSTM(semilla=1)
    modelo._centro = np.full(len(CARACTERISTICAS_DENSAS), 7.0)
    modelo._escala = np.ones(len(CARACTERISTICAS_DENSAS))

    fila = [float("nan")] * len(CARACTERISTICAS_DENSAS)
    secuencias = modelo._secuencias([fila])

    assert not np.isnan(secuencias).any()
    assert float(secuencias[0, 0, 0]) == 0.0  # 7 imputado, menos el centro 7


def test_la_lstm_aprende_una_tendencia_y_le_gana_a_la_persistencia():
    """La prueba que de verdad importa: que esté bien entrenada.

    Una línea base mal entrenada adula al modelo del proyecto, que es la dirección en la que no
    conviene equivocarse. Sobre una serie con tendencia clara, la recurrente tiene que quedar
    por debajo del error de repetir el valor de ahora.
    """
    pytest.importorskip("torch")
    import numpy as np

    from src.train.modelos import LSTM, Persistencia

    generador = np.random.default_rng(3)
    x, y = [], []
    for _ in range(600):
        inicio = float(generador.uniform(80, 200))
        pendiente = float(generador.uniform(-1.0, 1.0))
        fila = [0.0] * len(CARACTERISTICAS_DENSAS)
        for nombre, atras in (
            ("rezago_60", 60),
            ("rezago_30", 30),
            ("rezago_15", 15),
            ("rezago_10", 10),
            ("rezago_5", 5),
            ("mg_dl", 0),
        ):
            fila[CARACTERISTICAS_DENSAS.index(nombre)] = inicio - pendiente * atras
        x.append(fila)
        # El objetivo sigue la misma tendencia media hora más adelante.
        y.append(inicio + pendiente * 30)

    modelo = LSTM(semilla=7, epocas=120, paciencia=15)
    modelo.ajustar(x, y)

    error_lstm = _rmse(y, modelo.predecir(x))
    error_persistencia = _rmse(y, Persistencia().predecir(x))

    assert error_lstm < error_persistencia, (error_lstm, error_persistencia)


def test_la_lstm_es_repetible_con_la_misma_semilla():
    """Un resultado que no se puede repetir no sirve para un trabajo que se va a evaluar."""
    pytest.importorskip("torch")

    from src.train.modelos import LSTM

    x, y = _datos_de_juguete()
    predicciones = []
    for _ in range(2):
        modelo = LSTM(semilla=11, epocas=5, paciencia=5)
        modelo.ajustar(x, y)
        predicciones.append(modelo.predecir(x))

    assert predicciones[0] == predicciones[1]


def test_la_lstm_reporta_en_que_epoca_se_quedo():
    """Las épocas usadas van al manifiesto: «60 máximas» no dice nada si paró en la 12."""
    pytest.importorskip("torch")

    from src.train.modelos import LSTM

    x, y = _datos_de_juguete()
    modelo = LSTM(semilla=5, epocas=6, paciencia=6)
    modelo.ajustar(x, y)

    assert 1 <= modelo.hiperparametros["epocas_usadas"] <= 6
    assert modelo.hiperparametros["validacion"].startswith("último 20 %")


def test_la_lstm_no_predice_sin_ajustar():
    pytest.importorskip("torch")

    from src.train.modelos import LSTM

    with pytest.raises(RuntimeError, match="ajustar"):
        LSTM(semilla=1).predecir([[0.0] * len(CARACTERISTICAS_DENSAS)])


def _rmse(reales, predichos) -> float:
    return (sum((r - p) ** 2 for r, p in zip(reales, predichos, strict=True)) / len(reales)) ** 0.5
