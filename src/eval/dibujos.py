"""Las cuatro figuras del cuaderno de evaluación, y la rejilla de Clarke dibujada.

Son entregables del reporte, no adorno: la rejilla es el criterio principal del proyecto
(A+B ≥ 95 %) y una figura es la única forma de ver **dónde** cae el error, no solo cuánto mide.

## La decisión de este módulo: la rejilla se dibuja con el clasificador, no con rectas

La forma natural de dibujar una rejilla de Clarke es trazar a mano las rectas que separan las
zonas. Es también la forma de que el dibujo y el número dejen de coincidir sin que nadie lo
note: `src/eval/clarke.py` cambió dos veces en este proyecto —y la última vez movió 19 900
pares de B a C—, y unas rectas escritas aparte habrían seguido diciendo lo de antes. Una figura
que contradice a la métrica que acompaña es peor que ninguna figura, porque se publica.

Así que el fondo de la rejilla se pinta **evaluando `zona()` sobre una malla**. El dibujo no
puede discrepar del clasificador porque es el clasificador. El costo es una malla de unos
ochenta mil puntos, que tarda menos de un segundo.

## Lo que estas figuras respetan del resto del proyecto

- **Lo medido y lo predicho nunca se parecen**: línea continua contra línea punteada. Es la
  misma regla de la app, y por el mismo motivo: una persona medio dormida tiene que
  distinguirlas sin leer la leyenda. En el reporte, quien lee la figura puede estar igual de
  cansado.
- **El color nunca es el único portador de información.** Las cinco zonas van sombreadas de
  claro a oscuro **y** rotuladas con su letra, así que la figura se entiende impresa en gris.
- Cada figura lleva la **n** en el título. Un 96 % de A+B sobre doscientos pares y sobre
  doscientos mil no dicen lo mismo.
"""

from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.eval.clarke import ZONAS, zona
from src.eval.metrics import META_CLARKE_AB

# El rango válido de glucosa del proyecto (RF-05). El árbitro es
# `referencia/reglas_clinicas.py` de glucy-backend; aquí se usa para saber hasta dónde dibujar.
GLUCOSA_MIN = 20
GLUCOSA_MAX = 600

# Paso de la malla con que se pinta el fondo de la rejilla, en mg/dL. Uno, porque con dos las
# fronteras salen escalonadas y parecen un error del clasificador en lugar de un pixelado. Son
# 337 561 llamadas a `zona()` en el peor caso, que tardan menos de un segundo.
PASO_DE_LA_MALLA = 1

# Qué parte del dibujo tiene que ocupar un trozo de zona para que valga la pena rotularlo. Por
# debajo de esto la letra tapa más de lo que explica.
ROTULO_MINIMO = 0.012

# De claro a oscuro según la gravedad, para que la figura se lea en escala de grises. Son
# grises a propósito: la letra rotulada es la que dice qué zona es, y así ninguna figura del
# reporte depende de que se imprima en color.
GRISES = {"A": 0.97, "B": 0.88, "C": 0.72, "D": 0.58, "E": 0.40}


def _plt() -> Any:
    """Matplotlib se importa aquí dentro y no arriba.

    La integración continua instala solo pytest y ruff, y todo lo que sostiene las reglas
    clínicas está escrito con la biblioteca estándar. Un `import matplotlib` en la cabecera
    obligaría a instalar una rueda de cien megabytes para correr las pruebas de la rejilla.
    """
    import matplotlib

    matplotlib.use("Agg")  # sin ventana: esto corre en un cuaderno o en un servidor
    import matplotlib.pyplot as plt

    return plt


def rejilla_de_clarke(
    reales: Sequence[float],
    predichos: Sequence[float],
    *,
    titulo: str = "Rejilla de error de Clarke",
    hasta: int | None = None,
) -> Any:
    """La rejilla con los pares encima, el reparto por zona y el A+B contra su meta.

    `hasta` recorta el dibujo; por omisión llega a donde llegue el par más alto, redondeado
    hacia arriba, porque una rejilla dibujada hasta 600 con datos que no pasan de 300 deja los
    puntos apelotonados en una esquina y no se ve nada.
    """
    plt = _plt()
    from src.eval.clarke import reparto
    from src.eval.metrics import mae

    if not reales:
        raise ValueError("no hay pares que dibujar")

    tope = hasta or min(GLUCOSA_MAX, 50 * (int(max(max(reales), max(predichos))) // 50 + 2))
    malla = list(range(GLUCOSA_MIN, tope + 1, PASO_DE_LA_MALLA))
    fondo = [[GRISES[zona(x, y)] for x in malla] for y in malla]

    figura, eje = plt.subplots(figsize=(6.5, 6.5))
    eje.imshow(
        fondo,
        origin="lower",
        extent=(GLUCOSA_MIN, tope, GLUCOSA_MIN, tope),
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    eje.plot([GLUCOSA_MIN, tope], [GLUCOSA_MIN, tope], linewidth=0.8, color="black", alpha=0.5)
    eje.scatter(reales, predichos, s=8, color="#1f4e79", alpha=0.6, edgecolors="none")

    for letra, x, y in _donde_rotular(malla):
        eje.annotate(letra, (x, y), ha="center", va="center", fontsize=13, fontweight="bold")

    por_zona = reparto(reales, predichos)
    ab = por_zona["A"] + por_zona["B"]
    eje.set_title(
        f"{titulo}\n"
        f"A+B {ab:.1f} % (meta ≥ {META_CLARKE_AB:.0f} %) · "
        f"MAE {mae(reales, predichos):.1f} mg/dL · n = {len(reales)}\n"
        + " · ".join(f"{z} {por_zona[z]:.1f} %" for z in ZONAS)
    )
    eje.set_xlabel("Glucosa medida (mg/dL)")
    eje.set_ylabel("Glucosa predicha (mg/dL)")
    eje.set_xlim(GLUCOSA_MIN, tope)
    eje.set_ylim(GLUCOSA_MIN, tope)
    figura.tight_layout()
    return figura


def _donde_rotular(malla: Sequence[int]) -> list[tuple[str, float, float]]:
    """Dónde poner la letra de cada trozo de zona: en su punto **más interior**.

    No en el centro de masa, que fue lo primero que se escribió y salió mal: la zona E son dos
    esquinas opuestas y su centro de masa cae justo en medio del dibujo, encima de la zona A.
    Una figura que rotula «E» sobre la zona A es peor que una sin rótulos.

    Así que se hace en dos pasos, los dos sobre la misma malla con que se pintó el fondo:
    primero se parten las zonas en **trozos conexos** —la E son dos, la D son tres—, y después
    se busca en cada trozo el punto que queda más lejos de cualquier frontera. Ahí cabe la
    letra sin tocar la zona vecina.

    No se rotulan las esquirlas: un trozo que ocupa menos de la centésima parte del dibujo
    recibe una letra que tapa más de lo que explica.
    """
    ancho = len(malla)
    clases = [[zona(x, y) for x in malla] for y in malla]

    # Distancia de cada celda a la frontera más cercana, con un recorrido en anchura desde
    # todas las fronteras a la vez. Es una pasada sobre la malla, no una por celda.
    distancia = [[-1] * ancho for _ in range(ancho)]
    frente = deque()
    for fila in range(ancho):
        for columna in range(ancho):
            propia = clases[fila][columna]
            if fila in (0, ancho - 1) or columna in (0, ancho - 1):
                vecina_distinta = True
            else:
                vecina_distinta = any(
                    clases[fila + df][columna + dc] != propia
                    for df, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                )
            if vecina_distinta:
                distancia[fila][columna] = 0
                frente.append((fila, columna))
    while frente:
        fila, columna = frente.popleft()
        for df, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            f, c = fila + df, columna + dc
            if 0 <= f < ancho and 0 <= c < ancho and distancia[f][c] < 0:
                distancia[f][c] = distancia[fila][columna] + 1
                frente.append((f, c))

    rotulos: list[tuple[str, float, float]] = []
    visitadas = [[False] * ancho for _ in range(ancho)]
    minimo = ROTULO_MINIMO * ancho * ancho
    for inicio_f in range(ancho):
        for inicio_c in range(ancho):
            if visitadas[inicio_f][inicio_c]:
                continue
            letra = clases[inicio_f][inicio_c]
            celdas = 0
            mejor = (-1, inicio_f, inicio_c)
            pila = [(inicio_f, inicio_c)]
            visitadas[inicio_f][inicio_c] = True
            while pila:
                fila, columna = pila.pop()
                celdas += 1
                if distancia[fila][columna] > mejor[0]:
                    mejor = (distancia[fila][columna], fila, columna)
                for df, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    f, c = fila + df, columna + dc
                    if (
                        0 <= f < ancho
                        and 0 <= c < ancho
                        and not visitadas[f][c]
                        and clases[f][c] == letra
                    ):
                        visitadas[f][c] = True
                        pila.append((f, c))
            if celdas >= minimo:
                _, fila, columna = mejor
                rotulos.append((letra, malla[columna], malla[fila]))
    return rotulos


def medido_contra_predicho(
    minutos: Sequence[float],
    medido: Sequence[float],
    predicho: Sequence[float],
    *,
    titulo: str = "Lo medido y lo predicho",
    umbral_hipo: int = 70,
    umbral_hiper: int = 180,
) -> Any:
    """La serie medida y la predicha, una continua y la otra punteada.

    Es la figura que enseña lo que un número no enseña: **si el modelo llega tarde**. Un RMSE
    de 15 mg/dL puede ser una curva que sigue bien la forma o una que la sigue con diez minutos
    de retraso, y la segunda no sirve para anticipar nada.
    """
    plt = _plt()
    figura, eje = plt.subplots(figsize=(9, 4))

    eje.axhspan(umbral_hipo, umbral_hiper, color="black", alpha=0.05)
    eje.axhline(umbral_hipo, linewidth=0.9, color="black", alpha=0.45)
    eje.axhline(umbral_hiper, linewidth=0.9, color="black", alpha=0.45)
    eje.annotate(f"hipo {umbral_hipo}", (minutos[0], umbral_hipo), fontsize=8, va="bottom")
    eje.annotate(f"hiper {umbral_hiper}", (minutos[0], umbral_hiper), fontsize=8, va="bottom")

    eje.plot(minutos, medido, linewidth=2, color="#1f4e79", label="Medido")
    eje.plot(
        minutos, predicho, linewidth=2, linestyle=(0, (5, 3)), color="#c2410c", label="Predicho"
    )

    eje.set_title(f"{titulo} · n = {len(medido)}")
    eje.set_xlabel("Minutos")
    eje.set_ylabel("Glucosa (mg/dL)")
    eje.legend(loc="upper right")
    figura.tight_layout()
    return figura


def error_por_origen(mae_por_origen: dict[str, float], *, n_por_origen: dict[str, int]) -> Any:
    """El MAE desglosado por origen del dato, que es lo que destapa el sesgo del conjunto.

    Doce sujetos de OhioT1DM son doce personas con sensor. Si el error sale peor para las
    lecturas manuales y por foto, el sistema funciona peor **precisamente para quien no tiene
    sensor**, que es el usuario con menos recursos (riesgo R-2). Un promedio único lo esconde;
    esta figura lo pone enfrente.
    """
    plt = _plt()
    if not mae_por_origen:
        raise ValueError("no hay orígenes que dibujar")

    ordenados = sorted(mae_por_origen.items(), key=lambda par: -par[1])
    etiquetas = [f"{origen}\nn = {n_por_origen.get(origen, 0)}" for origen, _ in ordenados]
    valores = [valor for _, valor in ordenados]

    figura, eje = plt.subplots(figsize=(7, 0.62 * len(ordenados) + 1.5))
    barras = eje.barh(etiquetas, valores, color="#1f4e79", height=0.68)
    eje.bar_label(barras, fmt="%.1f mg/dL", fontsize=9, padding=4)
    eje.invert_yaxis()
    eje.set_xlabel("MAE (mg/dL)")
    eje.set_title(
        "Error por origen del dato\nSi el peor es el que no tiene sensor, hay que decirlo (R-2)"
    )
    eje.set_xlim(0, max(valores) * 1.25)
    figura.tight_layout()
    return figura


def contra_las_lineas_base(rmse_por_modelo: dict[str, float], *, el_del_proyecto: str) -> Any:
    """El modelo contra sus cinco líneas base, con la persistencia marcada.

    «Un modelo que no le gana a la persistencia no justifica su existencia» es una frase del
    protocolo, y esta es la figura que la vuelve comprobable de un vistazo. La persistencia se
    marca con una línea vertical: lo que quede a su derecha está perdiendo contra «dentro de
    media hora será igual que ahora».
    """
    plt = _plt()
    if el_del_proyecto not in rmse_por_modelo:
        raise ValueError(f"{el_del_proyecto} no está entre los modelos: {sorted(rmse_por_modelo)}")

    ordenados = sorted(rmse_por_modelo.items(), key=lambda par: par[1])
    etiquetas = [nombre for nombre, _ in ordenados]
    valores = [valor for _, valor in ordenados]
    # El del proyecto se distingue por el tono **y** por el asterisco de su etiqueta, para que
    # la figura siga diciendo cuál es en escala de grises.
    colores = ["#1f4e79" if n == el_del_proyecto else "#9ca3af" for n in etiquetas]
    etiquetas = [f"{n} *" if n == el_del_proyecto else n for n in etiquetas]

    figura, eje = plt.subplots(figsize=(7, 0.55 * len(ordenados) + 1.6))
    barras = eje.barh(etiquetas, valores, color=colores, height=0.68)
    eje.bar_label(barras, fmt="%.1f", fontsize=9, padding=4)
    eje.invert_yaxis()

    aviso = ""
    if (persistencia := rmse_por_modelo.get("persistencia")) is not None:
        eje.axvline(persistencia, linestyle=(0, (4, 3)), linewidth=1.4, color="black")
        perdiendo = [n for n, v in ordenados if v > persistencia and n != "persistencia"]
        if el_del_proyecto == "persistencia":
            # Pasa hoy, sin modelo entrenado: la figura enseña el listón, no un resultado.
            aviso = "\nTodavía no hay modelo: la única barra es la línea base"
        elif perdiendo:
            aviso = (
                "\nA la derecha de la línea no se le gana a «dentro de media hora será igual "
                f"que ahora»: {', '.join(perdiendo)}"
            )
        else:
            aviso = "\nTodos le ganan a la persistencia (la línea)"

    eje.set_xlabel("RMSE (mg/dL) · menos es mejor")
    eje.set_title(f"{el_del_proyecto} contra las líneas base del protocolo{aviso}", fontsize=10)
    eje.set_xlim(0, max(valores) * 1.3)
    figura.tight_layout()
    return figura


def guardar(figura: Any, ruta: Path | str, *, puntos_por_pulgada: int = 150) -> Path:
    """Guarda la figura y devuelve dónde quedó, creando la carpeta si falta.

    150 puntos por pulgada y no 300: el documento final se lee en pantalla y en PDF, y una
    figura de 300 multiplica el peso del archivo por cuatro sin que se vea mejor.
    """
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(ruta, dpi=puntos_por_pulgada, bbox_inches="tight")
    return ruta
