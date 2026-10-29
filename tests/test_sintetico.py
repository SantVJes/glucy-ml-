"""Pruebas del generador de series de juguete.

Lo que comprueban no es que las series sean realistas —no lo son, y el módulo lo dice en su
primera línea—, sino tres cosas de las que depende el resto de las pruebas:

1. Que sea **reproducible**, porque si no, una prueba que falla no se puede repetir.
2. Que las personas sean **distintas entre sí**, porque si fueran idénticas, repartir por
   persona y repartir al azar darían lo mismo y las pruebas de partición no comprobarían nada.
3. Que la serie tenga **hipoglucemias, hiperglucemias y huecos**, porque son los tres casos
   que el resto del repositorio existe para manejar.
"""

from datetime import datetime

from src.datos.sintetico import ORIGEN, conjunto, perfiles, registro


def test_la_misma_semilla_da_exactamente_la_misma_serie():
    """Una prueba que falla con datos irrepetibles no se puede arreglar."""
    p = perfiles(3, semilla=11)[0]
    primera = registro(p, dias=2, semilla=5)
    segunda = registro(p, dias=2, semilla=5)
    assert [f.mg_dl for f in primera.lecturas] == [f.mg_dl for f in segunda.lecturas]
    assert [f.ts for f in primera.lecturas] == [f.ts for f in segunda.lecturas]


def test_semillas_distintas_dan_series_distintas():
    p = perfiles(3)[0]
    a = [f.mg_dl for f in registro(p, dias=2, semilla=5).lecturas]
    b = [f.mg_dl for f in registro(p, dias=2, semilla=6).lecturas]
    assert a != b


def test_las_personas_son_distintas_entre_si():
    """Es lo que hace que repartir por persona signifique algo.

    Se mira el conjunto de los doce y no una pareja: dos personas cualesquiera pueden coincidir
    de promedio por casualidad, porque un nivel de base más alto se cancela con una
    sensibilidad a la insulina más alta. Lo que tiene que haber es **dispersión entre las
    doce**, y que ninguna serie sea idéntica a otra.
    """
    promedios = []
    series = []
    for perfil in perfiles(12):
        valores = [f.mg_dl for f in registro(perfil, dias=2).lecturas]
        promedios.append(sum(valores) / len(valores))
        series.append(tuple(valores))
    assert max(promedios) - min(promedios) > 20.0
    assert len(set(series)) == 12


def test_hay_doce_personas_por_omision():
    """Las mismas que OhioT1DM, para que LOSO trabaje con el mismo número de pliegues."""
    assert len(perfiles()) == 12


def test_las_lecturas_salen_ordenadas_y_del_sensor():
    r = registro(perfiles(1)[0], dias=1)
    assert [f.ts for f in r.lecturas] == sorted(f.ts for f in r.lecturas)
    assert all(f.del_sensor for f in r.lecturas)


def test_la_serie_tiene_huecos():
    """El conjunto real los tiene, y un generador perfecto esconde los errores de rezago."""
    r = registro(perfiles(1)[0], dias=3, probabilidad_de_hueco=0.05)
    esperadas = 3 * 24 * 60 // 5
    assert len(r.lecturas) < esperadas


def test_sin_probabilidad_de_hueco_la_serie_es_completa():
    r = registro(perfiles(1)[0], dias=1, probabilidad_de_hueco=0.0)
    assert len(r.lecturas) == 24 * 60 // 5


def test_la_mezcla_tiene_hipoglucemias_e_hiperglucemias():
    """Sin valores por debajo de 70 no se puede probar la sensibilidad a hipoglucemia, que es
    la métrica que más importa; sin valores por encima de 180 no se ejercita la zona C de la
    rejilla de Clarke.
    """
    valores = [f.mg_dl for r in conjunto(12, dias=3) for f in r.lecturas]
    assert any(v < 70 for v in valores)
    assert any(v > 180 for v in valores)
    en_rango = sum(1 for v in valores if 70 <= v <= 180) / len(valores)
    assert 0.5 < en_rango < 0.9


def test_todos_los_valores_estan_en_un_rango_fisiologico():
    """Nada por debajo de 20 ni por encima de 600: no pasaría la validación del proyecto."""
    valores = [f.mg_dl for r in conjunto(6, dias=2) for f in r.lecturas]
    assert min(valores) >= 20
    assert max(valores) <= 600


def test_cada_dia_trae_tres_comidas_con_su_dosis():
    r = registro(perfiles(1)[0], dias=4)
    assert len(r.eventos["meal"]) == 12
    assert len(r.eventos["bolus"]) == 12


def test_las_comidas_y_las_dosis_salen_ordenadas():
    r = registro(perfiles(1)[0], dias=3)
    for seccion in ("meal", "bolus"):
        marcas = [e.ts for e in r.eventos[seccion]]
        assert marcas == sorted(marcas)


def test_las_comidas_traen_gramos_y_las_dosis_unidades():
    r = registro(perfiles(1)[0], dias=2)
    assert all(0 < e.valor <= 300 for e in r.eventos["meal"])
    assert all(0 < e.valor <= 50 for e in r.eventos["bolus"])


def test_el_registro_tiene_la_forma_que_devuelve_el_lector_de_verdad():
    """Así el resto del repositorio no sabe si los datos son reales o de juguete, y da igual."""
    r = registro(perfiles(1)[0], dias=1)
    assert r.sujeto == perfiles(1)[0].sujeto
    assert isinstance(r.lecturas, tuple)
    assert isinstance(r.eventos, dict)


def test_el_conjunto_entero_se_puede_recorrer():
    registros = list(conjunto(4, dias=1))
    assert len(registros) == 4
    assert len({r.sujeto for r in registros}) == 4


def test_el_inicio_se_puede_fijar():
    cuando = datetime(2025, 6, 1, 0, 0)
    r = registro(perfiles(1)[0], dias=1, inicio=cuando)
    assert r.lecturas[0].ts == cuando


def test_el_origen_esta_declarado_como_sintetico():
    """Es lo que acaba en el manifiesto del artefacto para que sus métricas no se reporten."""
    assert ORIGEN == "sintetico"
