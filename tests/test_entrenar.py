"""Pruebas del bucle de entrenamiento.

Usan un modelo de mentiras en lugar de XGBoost, así que **corren en la integración continua**.
Es a propósito: lo que hay que probar aquí no es si un modelo aprende, sino que el bucle no
filtre información entre pliegues y que agregue las métricas como dice que las agrega. Un
modelo real haría estas pruebas lentas y tapadas por su propio ruido.
"""

import math

import pytest

from src.features.features import CARACTERISTICAS_DENSAS
from src.features.ventanas import Fila
from src.train.entrenar import evaluar_con_loso, evaluar_con_loso_simple, resumen
from src.train.modelos import ExtrapolacionLineal, Persistencia

I_VALOR = CARACTERISTICAS_DENSAS.index("mg_dl")


def caracteristicas(valor: float, **sobre: float) -> dict[str, float]:
    base = {nombre: 0.0 for nombre in CARACTERISTICAS_DENSAS}
    base["mg_dl"] = valor
    base["rezago_30"] = valor
    return {**base, **sobre}


def filas_de(sujeto: str, valores, objetivos=None) -> list[Fila]:
    objetivos = objetivos if objetivos is not None else valores
    return [
        Fila(
            sujeto=sujeto,
            ts_segundos=float(i * 300),
            caracteristicas=caracteristicas(float(v)),
            objetivo=float(o),
        )
        for i, (v, o) in enumerate(zip(valores, objetivos, strict=True))
    ]


def tres_sujetos() -> list[Fila]:
    """Tres sujetos cuyo objetivo es siempre el valor más 5, para que un modelo «perfecto» exista.

    C vive en hipoglucemia a propósito: sin un solo episodio real por debajo de 70, la
    sensibilidad queda en `None` y la mitad de estas pruebas no comprobaría nada.
    """
    return (
        filas_de("A", [100, 110, 120, 130], [105, 115, 125, 135])
        + filas_de("B", [140, 150, 160, 170], [145, 155, 165, 175])
        + filas_de("C", [60, 65, 70, 75], [65, 70, 75, 80])
    )


class ModeloEspia:
    """Un modelo que anota con qué valores lo entrenaron y predice la media de lo que vio.

    Sirve para comprobar que **cada pliegue recibe un modelo nuevo** y que el sujeto de prueba
    nunca aparece en el entrenamiento de su propio pliegue.
    """

    nombre = "espia"
    algoritmo = "espia"
    vistos: list[list[float]] = []

    def __init__(self) -> None:
        self.entrenado_con: list[float] = []
        ModeloEspia.vistos.append(self.entrenado_con)

    @property
    def hiperparametros(self) -> dict[str, float]:
        return {}

    def ajustar(self, x, y) -> None:
        self.entrenado_con.extend(float(fila[I_VALOR]) for fila in x)

    def predecir(self, x) -> list[float]:
        media = sum(self.entrenado_con) / len(self.entrenado_con) if self.entrenado_con else 0.0
        return [media] * len(x)


@pytest.fixture(autouse=True)
def limpiar_espia():
    ModeloEspia.vistos = []
    yield
    ModeloEspia.vistos = []


# --- Que no se filtre nada entre pliegues ----------------------------------------------------


def test_cada_pliegue_recibe_un_modelo_nuevo():
    """Reutilizar el objeto lo seguiría entrenando con los pliegues anteriores.

    Eso es fuga entre pliegues: el modelo del último pliegue habría visto a los doce sujetos y
    el resultado saldría inflado sin que nada falle.
    """
    evaluar_con_loso(
        tres_sujetos(),
        juego="denso",
        horizonte_min=30,
        construir_modelo=ModeloEspia,
    )
    assert len(ModeloEspia.vistos) == 3
    assert all(len(v) == 8 for v in ModeloEspia.vistos)  # 3 sujetos - 1, por 4 filas


def test_el_sujeto_de_prueba_nunca_entra_a_su_propio_entrenamiento():
    evaluar_con_loso(
        tres_sujetos(),
        juego="denso",
        horizonte_min=30,
        construir_modelo=ModeloEspia,
    )
    # El pliegue de C entrena solo con A y B, cuyos valores están todos por encima de 99.
    de_c = [v for v in ModeloEspia.vistos if 60.0 not in v]
    assert len(de_c) == 1
    assert min(de_c[0]) >= 100.0


# --- Cómo se agregan las métricas -------------------------------------------------------------


def test_las_metricas_se_agregan_sobre_todas_las_predicciones():
    """Y no promediando los números por pliegue.

    Promediar porcentajes de muestras de distinto tamaño los pondera mal: un sujeto con 400
    filas acabaría pesando lo mismo que uno con 4 000.
    """
    filas = filas_de("A", [100] * 10) + filas_de("B", [200] * 2)
    corrida = evaluar_con_loso(filas, juego="denso", horizonte_min=30, construir_modelo=ModeloEspia)
    assert corrida.resultado.n == 12


def test_hay_un_resultado_por_sujeto():
    """Un promedio esconde a la persona para la que el sistema funciona peor (riesgo R-2)."""
    corrida = evaluar_con_loso(
        tres_sujetos(), juego="denso", horizonte_min=30, construir_modelo=ModeloEspia
    )
    assert sorted(corrida.por_sujeto) == ["A", "B", "C"]
    assert all(r.n == 4 for r in corrida.por_sujeto.values())


def test_la_corrida_conserva_el_horizonte_y_los_sujetos():
    corrida = evaluar_con_loso(
        tres_sujetos(),
        juego="denso",
        horizonte_min=60,
        construir_modelo=ModeloEspia,
        nombre="m2",
    )
    assert corrida.horizonte_min == 60
    assert corrida.sujetos == ("A", "B", "C")
    assert corrida.modelo == "m2"


def test_sin_filas_revienta():
    with pytest.raises(ValueError, match="no hay filas"):
        evaluar_con_loso([], juego="denso", horizonte_min=30, construir_modelo=ModeloEspia)


# --- Las líneas base, en los mismos pliegues --------------------------------------------------


def test_las_lineas_base_se_miden_sobre_las_mismas_filas():
    """Es el punto entero: comparar contra un número de otra corrida no es comparar."""
    corrida = evaluar_con_loso(
        tres_sujetos(),
        juego="denso",
        horizonte_min=30,
        construir_modelo=ModeloEspia,
        construir_lineas_base=lambda: {"persistencia": Persistencia()},
    )
    assert corrida.lineas_base["persistencia"].n == corrida.resultado.n


def test_las_predicciones_nan_se_descartan_y_el_n_lo_delata():
    """La extrapolación produce `NaN` al empezar la serie. El `n` dice sobre cuántas se midió."""
    filas = tres_sujetos()
    for i, f in enumerate(filas):
        if i % 4 == 0:
            f.caracteristicas["rezago_30"] = math.nan
    corrida = evaluar_con_loso(
        filas,
        juego="denso",
        horizonte_min=30,
        construir_modelo=ModeloEspia,
        construir_lineas_base=lambda: {"extrapolacion": ExtrapolacionLineal(30)},
    )
    assert corrida.lineas_base["extrapolacion"].n < corrida.resultado.n


# --- La pregunta que decide si el modelo sirve ------------------------------------------------


def test_le_gana_a_la_persistencia_exige_mejor_rmse_y_no_peor_a_mas_b():
    """Un RMSE mejor con más predicciones en zonas peligrosas no es mejor para una persona."""
    corrida = evaluar_con_loso(
        tres_sujetos(),
        juego="denso",
        horizonte_min=30,
        construir_modelo=ModeloEspia,
        construir_lineas_base=lambda: {"persistencia": Persistencia()},
    )
    # El espía predice la media del entrenamiento, que es malísima: la persistencia le gana.
    assert not corrida.le_gana_a_la_persistencia


def test_sin_persistencia_no_se_puede_afirmar_que_le_gana():
    """Si no se midió, la respuesta es «no», no «sí por omisión»."""
    corrida = evaluar_con_loso(
        tres_sujetos(), juego="denso", horizonte_min=30, construir_modelo=ModeloEspia
    )
    assert not corrida.le_gana_a_la_persistencia


def test_un_modelo_perfecto_le_gana_a_la_persistencia():
    class Perfecto(ModeloEspia):
        nombre = "perfecto"

        def __init__(self) -> None:
            super().__init__()
            self._y: list[float] = []

        def ajustar(self, x, y) -> None:
            super().ajustar(x, y)
            self._y = list(y)

        def predecir(self, x):
            # Los objetivos son el valor más 5 en todos los sujetos de prueba.
            return [float(fila[I_VALOR]) + 5.0 for fila in x]

    corrida = evaluar_con_loso(
        tres_sujetos(),
        juego="denso",
        horizonte_min=30,
        construir_modelo=Perfecto,
        construir_lineas_base=lambda: {"persistencia": Persistencia()},
    )
    assert corrida.resultado.rmse == pytest.approx(0.0)
    assert corrida.le_gana_a_la_persistencia


# --- El objetivo que no es una glucosa -------------------------------------------------------


def test_un_objetivo_en_minutos_se_mide_solo_con_rmse_y_mae():
    """«Zona A» sobre una cantidad de minutos no significa nada.

    Pasar minutos por `evaluar` devolvería números con nombre clínico y sin sentido, que es
    peor que no tenerlos.
    """
    filas = tres_sujetos()
    resultado, por_sujeto = evaluar_con_loso_simple(
        filas,
        juego="denso",
        objetivo=lambda f: 75.0,
        construir_modelo=ModeloEspia,
        unidad="min",
    )
    assert resultado.unidad == "min"
    assert resultado.n == len(filas)
    assert not hasattr(resultado, "clarke_ab")
    assert sorted(por_sujeto) == ["A", "B", "C"]


# --- El cuadro que se pega en el documento ----------------------------------------------------


def test_el_resumen_trae_el_modelo_las_lineas_base_y_el_veredicto():
    corrida = evaluar_con_loso(
        tres_sujetos(),
        juego="denso",
        horizonte_min=30,
        construir_modelo=ModeloEspia,
        construir_lineas_base=lambda: {"persistencia": Persistencia()},
        nombre="m1",
    )
    texto = resumen(corrida)
    assert "m1" in texto
    assert "persistencia" in texto
    assert "Le gana a la persistencia" in texto
    assert "RMSE" in texto


def test_el_resumen_dice_no_se_midio_cuando_no_hubo_hipoglucemias():
    """Nunca un cero, que se leería como «no detectó ninguna»."""
    sin_hipos = filas_de("A", [120, 130, 140, 150]) + filas_de("B", [160, 170, 180, 190])
    corrida = evaluar_con_loso(
        sin_hipos, juego="denso", horizonte_min=30, construir_modelo=ModeloEspia
    )
    assert corrida.resultado.sensibilidad_hipo is None
    assert "no se midió" in resumen(corrida)
