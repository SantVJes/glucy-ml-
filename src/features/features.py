"""Los dos juegos de características de RF-24, con **los nombres y el orden del backend**.

`app/services/pipeline/caracteristicas.py` de `glucy-backend` es la autoridad sobre estas
listas, y lo dice en su propio encabezado: «glucy-ml copia estas tuplas, no las reinventa».
Aquí se copian, no se derivan.

**Por qué esto es lo más delicado del repositorio.** El artefacto entrenado recibe un vector de
números en un orden. Si aquí se llaman o se ordenan distinto que en el pipeline que lo alimenta
en producción, la predicción se calcula con las columnas cambiadas de lugar: salen números
**plausibles y equivocados**, el modelo no falla, la API responde 200 y nadie se entera hasta
que las métricas de producción no se parecen a las del cuaderno. `GET /predict` del backend se
defiende comparando la lista declarada en el manifiesto contra la suya y negándose a cargar el
artefacto si no coinciden; este archivo es el otro lado de esa comparación, y por eso el
entrenamiento **escribe la lista en el manifiesto** en lugar de dejarla implícita.

`tests/test_caracteristicas.py` fija las dos tuplas carácter por carácter. Si alguien agrega,
quita o mueve una, la prueba lo dice. Es una prueba de valores literales a propósito: los dos
repositorios no se pueden importar entre sí, así que la única red posible es que el cambio sea
imposible de hacer sin verlo.

Las reglas que van con las listas, también del backend:

- **Lo que falta es `NaN`, nunca cero.** Un rezago de 60 minutos que no existe porque la serie
  empieza ahí no es una glucosa de cero; XGBoost trata el `NaN` como información —aprende para
  qué lado mandar la fila— y el cero como un valor real que arrastra el promedio hacia abajo.
- **Ninguna característica sale de una comida que la persona no registró** (D-9, RF-37).
- **Sin `actividad` ni `sueno` en el juego denso** (C-10, C-11): el teléfono no las captura y
  el servidor no tiene endpoint para el contexto, así que en producción siempre faltarían.
  Entrenar con una variable que en producción nunca llega es peor que no tenerla: el modelo le
  da peso y luego nunca lo recibe. **Esto vale aunque OhioT1DM sí traiga ejercicio y sueño**,
  que es la tentación obvia al tener el conjunto delante.
- **El peso no entra.** El conjunto lo trae como relleno (99 para todos) y, dentro de una misma
  persona, es constante en un horizonte de 30 minutos.
"""

import math
from collections.abc import Sequence

# Copiadas de app/services/pipeline/caracteristicas.py de glucy-backend. No derivar, no ordenar
# por programa, no generar: copiadas.
CARACTERISTICAS_DENSAS: tuple[str, ...] = (
    "mg_dl", "rezago_5", "rezago_10", "rezago_15", "rezago_30", "rezago_60",
    "derivada_1", "derivada_2", "media_30", "media_60",
    "cob", "iob", "hora_sin", "hora_cos", "dia_semana",
)  # fmt: skip
CARACTERISTICAS_DISPERSAS: tuple[str, ...] = (
    "mg_dl", "antiguedad_min", "carbs_g", "dosis_ui",
    "cob", "iob", "hora_sin", "hora_cos", "contexto_comida", "promedio_7d",
)  # fmt: skip

# `contexto_comida` es categórica: un entero fijo por valor del enum, y «sin contexto» tiene su
# propio código en lugar de un cero disfrazado. **Cambiar un número de aquí obliga a
# reentrenar**, porque el artefacto aprendió estos códigos y no sus nombres.
CODIGOS_CONTEXTO: dict[str | None, int] = {
    "en_ayunas": 0,
    "antes_de_comer": 1,
    "dos_horas_despues": 2,
    "antes_de_dormir": 3,
    "otro": 4,
    None: 5,
}

REZAGOS_MIN = (5, 10, 15, 30, 60)
MINUTOS_POR_DIA = 1440
FALTA = math.nan

JUEGOS = {"denso": CARACTERISTICAS_DENSAS, "disperso": CARACTERISTICAS_DISPERSAS}


def hora_ciclica(minutos_del_dia: float) -> tuple[float, float]:
    """La hora del día como seno y coseno, no como un número de 0 a 23.

    Las 23:50 y las 00:10 están a veinte minutos, pero como números están a 23 horas. Un
    modelo de árboles puede aprender a partir el día en trozos, pero no que los extremos se
    tocan, y la medianoche es justo cuando pasa lo que más importa: la hipoglucemia nocturna.
    """
    angulo = 2 * math.pi * (minutos_del_dia % MINUTOS_POR_DIA) / MINUTOS_POR_DIA
    return math.sin(angulo), math.cos(angulo)


def codigo_contexto(contexto: str | None) -> float:
    """El código de un contexto de comida, o el de «sin contexto».

    Un contexto desconocido es un error y no un sexto código: significa que el enum cambió de
    un lado y no del otro, y descubrirlo aquí cuesta un `ValueError` mientras descubrirlo en
    producción cuesta una columna mal alimentada.
    """
    if contexto not in CODIGOS_CONTEXTO:
        raise ValueError(
            f"contexto de comida desconocido: {contexto!r}. "
            f"Los válidos son {sorted(k for k in CODIGOS_CONTEXTO if k)} o None"
        )
    return float(CODIGOS_CONTEXTO[contexto])


def a_vector(fila: dict[str, float], juego: str) -> list[float]:
    """De un diccionario de características al vector que recibe el modelo, **en el orden de la
    tupla del juego**.

    Exige que estén todas las llaves y ninguna de más. Una llave que falta no se rellena con
    cero ni se ignora: sería una columna desplazada, que es el error que este archivo existe
    para evitar. Que falte el **valor** es legítimo —eso es `NaN` y el modelo lo entiende—;
    que falte la **llave** no.
    """
    if juego not in JUEGOS:
        raise ValueError(f"juego desconocido: {juego!r}. Los válidos son 'denso' y 'disperso'")
    nombres = JUEGOS[juego]
    faltan = set(nombres) - set(fila)
    sobran = set(fila) - set(nombres)
    if faltan or sobran:
        raise ValueError(
            f"el juego {juego!r} no cuadra: faltan {sorted(faltan)}, sobran {sorted(sobran)}"
        )
    return [float(fila[nombre]) for nombre in nombres]


def a_matriz(filas: Sequence[dict[str, float]], juego: str) -> list[list[float]]:
    """Varias filas al mismo orden. Se construye fila por fila con `a_vector` a propósito, para
    que una fila mal armada en la número ochocientas falle igual que la primera.
    """
    return [a_vector(fila, juego) for fila in filas]


def manifiesto_de_caracteristicas(juego: str) -> list[str]:
    """La lista que va al manifiesto del artefacto, para que `GET /predict` la compare.

    Es una copia y no la tupla misma: el manifiesto se serializa a JSON y una tupla compartida
    por referencia invita a que alguien la ordene en el sitio y mueva el contrato sin querer.
    """
    if juego not in JUEGOS:
        raise ValueError(f"juego desconocido: {juego!r}. Los válidos son 'denso' y 'disperso'")
    return list(JUEGOS[juego])
