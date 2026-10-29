"""Pruebas del reparto de datos, que es lo que más fácilmente invalida un resultado entero.

Son pruebas sobre reglas y no sobre cálculos: lo que comprueban es que un sujeto no pueda
quedar en los dos lados y que el futuro no entrene. Las dos fallas se ven igual de bien que
una buena partición mirando las métricas, y por eso tienen que fallar aquí.
"""

from dataclasses import dataclass

import pytest

from src.eval.particion import (
    Reparto,
    hay_fuga_temporal,
    partir_en_el_tiempo,
    partir_por_sujeto,
    pliegues_loso,
    sujetos_de,
)


@dataclass(frozen=True)
class FilaDePrueba:
    sujeto: str
    orden: float


def filas(*pares: tuple[str, float]) -> list[FilaDePrueba]:
    return [FilaDePrueba(sujeto=s, orden=t) for s, t in pares]


DOCE = [FilaDePrueba(sujeto=f"{559 + i}", orden=float(t)) for i in range(12) for t in range(5)]


# --- Por persona, nunca al azar --------------------------------------------------------------


def test_un_sujeto_de_prueba_no_deja_ni_una_fila_en_entrenamiento():
    """Es la regla entera: si una sola fila de esa persona se cuela, el modelo la memoriza."""
    datos = filas(("a", 1), ("a", 2), ("b", 1), ("c", 1))
    r = partir_por_sujeto(datos, ["a"])
    assert {f.sujeto for f in r.entrenamiento} == {"b", "c"}
    assert {f.sujeto for f in r.prueba} == {"a"}
    assert r.sujetos_prueba == ("a",)
    assert r.sujetos_entrenamiento == ("b", "c")


def test_un_reparto_con_un_sujeto_en_los_dos_lados_no_se_puede_construir():
    """La comprobación está en el propio `Reparto` y no en la función que lo arma.

    Así cubre también un reparto armado a mano en un cuaderno, que es justo donde alguien va a
    improvisar uno.
    """
    with pytest.raises(ValueError, match="están en los dos lados"):
        Reparto(
            entrenamiento=[],
            prueba=[],
            sujetos_entrenamiento=("a", "b"),
            sujetos_prueba=("b",),
        )


def test_un_sujeto_de_prueba_mal_escrito_revienta():
    """Un nombre mal escrito daría un conjunto de prueba vacío y una métrica sobre nada."""
    datos = filas(("a", 1), ("b", 1))
    with pytest.raises(ValueError, match="no están en los datos"):
        partir_por_sujeto(datos, ["z"])


def test_no_se_puede_dejar_el_entrenamiento_vacio():
    datos = filas(("a", 1), ("b", 1))
    with pytest.raises(ValueError, match="ningún sujeto para entrenar"):
        partir_por_sujeto(datos, ["a", "b"])


def test_hay_que_nombrar_al_menos_un_sujeto_de_prueba():
    with pytest.raises(ValueError, match="al menos un sujeto"):
        partir_por_sujeto(filas(("a", 1)), [])


def test_los_sujetos_salen_ordenados_y_no_en_orden_de_aparicion():
    """Un reparto que cambia según cómo se leyeron los archivos no se puede repetir."""
    datos = filas(("c", 1), ("a", 1), ("b", 1), ("a", 2))
    assert sujetos_de(datos) == ("a", "b", "c")


# --- LOSO -----------------------------------------------------------------------------------


def test_loso_da_un_pliegue_por_sujeto():
    pliegues = pliegues_loso(DOCE)
    assert len(pliegues) == 12
    assert [p.sujetos_prueba for p in pliegues] == [(f"{559 + i}",) for i in range(12)]


def test_cada_pliegue_de_loso_entrena_con_once_y_prueba_con_uno():
    for pliegue in pliegues_loso(DOCE):
        assert len(pliegue.sujetos_entrenamiento) == 11
        assert len(pliegue.sujetos_prueba) == 1
        assert not set(pliegue.sujetos_entrenamiento) & set(pliegue.sujetos_prueba)


def test_cada_sujeto_es_de_prueba_exactamente_una_vez():
    vistos = [p.sujetos_prueba[0] for p in pliegues_loso(DOCE)]
    assert sorted(vistos) == sorted(sujetos_de(DOCE))


def test_loso_devuelve_una_lista_y_no_un_generador():
    """Los pliegues se recorren varias veces: una por modelo y una por línea base.

    Un generador agotado daría cero pliegues en la segunda pasada sin que nada falle, y la
    línea base saldría medida sobre nada.
    """
    pliegues = pliegues_loso(DOCE)
    assert len(list(pliegues)) == 12
    assert len(list(pliegues)) == 12


def test_loso_con_un_solo_sujeto_revienta():
    with pytest.raises(ValueError, match="al menos dos sujetos"):
        pliegues_loso(filas(("a", 1), ("a", 2)))


# --- El pasado entrena, el futuro prueba -----------------------------------------------------


def test_el_corte_temporal_deja_el_pasado_en_entrenamiento():
    datos = filas(*[("a", float(t)) for t in range(10)])
    r = partir_en_el_tiempo(datos, 0.8)
    assert [f.orden for f in r["entrenamiento"]] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert [f.orden for f in r["prueba"]] == [8, 9]


def test_el_corte_es_por_sujeto_y_no_sobre_el_conjunto_entero():
    """Los sujetos no comparten calendario: cada uno aportó sus ocho semanas cuando le tocó.

    Con un corte por fecha global, el sujeto `tarde` quedaría entero en prueba y `pronto`
    entero en entrenamiento, que es una partición por persona disfrazada de partición temporal.
    """
    datos = filas(
        *[("pronto", float(t)) for t in range(10)],
        *[("tarde", float(1000 + t)) for t in range(10)],
    )
    r = partir_en_el_tiempo(datos, 0.8)
    assert {f.sujeto for f in r["entrenamiento"]} == {"pronto", "tarde"}
    assert {f.sujeto for f in r["prueba"]} == {"pronto", "tarde"}


def test_el_corte_no_baraja_ni_depende_del_orden_de_entrada():
    """Se ordena por `orden` antes de cortar, así que las filas pueden llegar desordenadas."""
    desordenadas = filas(("a", 5), ("a", 1), ("a", 4), ("a", 2), ("a", 3))
    r = partir_en_el_tiempo(desordenadas, 0.6)
    assert [f.orden for f in r["entrenamiento"]] == [1, 2, 3]
    assert [f.orden for f in r["prueba"]] == [4, 5]


def test_un_sujeto_con_muy_pocas_filas_revienta():
    """Con una sola fila el corte deja un lado vacío, y eso se avisa.

    Devolver un conjunto de prueba de cero filas daría métricas sobre nada. Con dos filas y
    una fracción de 0.8 el corte sí alcanza —una y una—, así que el caso límite de verdad es
    el sujeto que aportó una sola lectura.
    """
    with pytest.raises(ValueError, match="un lado queda vacío"):
        partir_en_el_tiempo(filas(("a", 1), ("b", 1), ("b", 2)), 0.8)


def test_con_dos_filas_el_corte_alcanza_para_las_dos_partes():
    r = partir_en_el_tiempo(filas(("a", 1), ("a", 2)), 0.8)
    assert [f.orden for f in r["entrenamiento"]] == [1]
    assert [f.orden for f in r["prueba"]] == [2]


def test_una_fraccion_fuera_de_rango_revienta():
    for fraccion in (0.0, 1.0, -0.2, 1.5):
        with pytest.raises(ValueError, match="entre 0 y 1"):
            partir_en_el_tiempo(filas(("a", 1), ("a", 2)), fraccion)


# --- La comprobación de fuga ----------------------------------------------------------------


def test_un_corte_correcto_no_tiene_fuga():
    datos = filas(*[("a", float(t)) for t in range(10)])
    r = partir_en_el_tiempo(datos, 0.8)
    assert hay_fuga_temporal(r["entrenamiento"], r["prueba"]) == []


def test_la_fuga_se_detecta_y_dice_en_quien():
    """Entrenar con el futuro no existe en producción: a las 14:00 no hay lecturas de las 15:00."""
    entrenamiento = filas(("a", 100), ("b", 1))
    prueba = filas(("a", 50), ("b", 2))
    assert hay_fuga_temporal(entrenamiento, prueba) == ["a"]


def test_un_empate_exacto_cuenta_como_fuga():
    """La misma marca de tiempo en los dos lados es la misma fila contada dos veces."""
    assert hay_fuga_temporal(filas(("a", 10)), filas(("a", 10))) == ["a"]


def test_un_sujeto_que_solo_esta_en_un_lado_no_es_fuga():
    """Es lo normal en LOSO: el sujeto de prueba no aparece en entrenamiento."""
    assert hay_fuga_temporal(filas(("a", 100)), filas(("b", 1))) == []
