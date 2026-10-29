"""La rejilla de error de Clarke, que es el criterio principal del proyecto.

Clasifica el error de una predicción por su **consecuencia clínica** y no por su tamaño: un
error de 30 mg/dL cerca de 70 puede mandar a alguien a una hipoglucemia, y el mismo error
cerca de 200 no cambia ninguna decisión. Meta del proyecto: **≥ 95 % de las predicciones en
A+B**.

Las cinco zonas, en los términos del artículo original:

| Zona | Qué significa |
| --- | --- |
| A | clínicamente exacta: lleva a la misma decisión que el valor real |
| B | se desvía, pero lleva a un tratamiento benigno o a ninguno |
| C | lleva a **sobrecorregir** una glucosa que estaba aceptable |
| D | lleva a **no detectar** una glucosa que estaba fuera de rango |
| E | lleva a un tratamiento **contrario** al que hacía falta |

## Por qué este archivo no copia la referencia del backend

`referencia/reglas_clinicas.py` de `glucy-backend` es el árbitro de las reglas clínicas y este
repositorio tiene que dar sus mismos números. Su propio `docstring` de `zona_clarke` advertía
que la implementación seguía «la formulación habitual» y que **convenía cotejarla antes de
reportar métricas**.

Se cotejó (29/10/2026). Se comparó la referencia del backend contra el puerto de Python de la
implementación de MATLAB File Exchange, par por par sobre los 337 561 pares enteros de 20 a
600 mg/dL. **No coinciden en 29 690 pares, en tres familias**, y no todas van en la misma
dirección:

| # | Pares | Con un real… y un predicho… | El backend dice | El puerto público dice |
| --- | --- | --- | --- | --- |
| 1 | 28 105 | real 181–290, predicho ≥ real + 110 | B | **C** |
| 2 | 361 | real ≥ 240, predicho exactamente 180 | B | **D** |
| 3 | 1 224 | real 59–70, predicho 71–180 | **D** | B |

Casos concretos: (200, 320) para la 1, (300, 180) para la 2, (60, 120) para la 3.

Qué es cada una:

1. El backend acota la zona C por arriba en 180 en lugar de 290. Un real de 200 predicho como
   320 invita a sobrecorregir: eso es C, no un error benigno.
2. El backend compara con `70 < p < 180` donde el puerto usa `70 ≤ p ≤ 180`. Un `<` donde va
   un `<=`.
3. Aquí es **el puerto el más indulgente**. Su condición para la zona D baja es
   `p ≥ (7/5)·real + 110`, que con un real de 60 pide un predicho de 194, y como arriba de 180
   ya manda la zona E, esa rama nunca se activa: un real de 60 predicho como 120 le sale **B**.
   Pero un real de 60 predicho como 120 es literalmente «no detectar una hipoglucemia», que es
   la definición de la zona D y **el evento central de Glucy**. La fórmula del puerto parece una
   transcripción equivocada; el backend aquí tiene razón.

**Este archivo toma en las tres la lectura más severa**: C en la 1, D en la 2 y D en la 3. Así
que difiere del backend en las dos primeras y del puerto público en la tercera.

Importa porque B cuenta para el A+B que el proyecto promete en ≥ 95 %: cada vez que una
formulación llama B a lo que otra llama C o D, **infla justo la cifra que decide si el modelo
sirve**. Elegir la lectura severa es la decisión conservadora: si al cotejar contra el artículo
original resultara que la indulgente es la correcta, el proyecto habría reportado una métrica
**peor** que la real, que es el error que se puede defender. Al contrario, no.

La referencia del backend se corrige para las dos primeras, para que los dos repositorios den
el mismo número.

`tests/test_clarke.py` deja escritos los tres casos concretos, así que volver a cualquiera de
las otras dos formulaciones hace fallar una prueba en lugar de mover la métrica en silencio.

**Lo que sigue pendiente** es cotejar contra el artículo original, que no está accesible en
línea sin suscripción. Mientras no se haga, el documento final tiene que decir qué formulación
se usó, no solo «rejilla de Clarke». El puerto público con el que se cotejó:
https://github.com/suetAndTie/ClarkeErrorGrid.
"""

from collections.abc import Sequence

ZONAS = ("A", "B", "C", "D", "E")

# El límite de 20 % que define la zona A, y los dos umbrales clínicos que definen el resto.
TOLERANCIA_A = 0.2
HIPO = 70
HIPER = 180


def zona(real: float, predicho: float) -> str:
    """La zona de un par (real, predicho). **El orden de las condiciones decide el resultado**:
    las zonas se solapan, y se evalúan de la más benigna a la más grave por diseño.

    No valida el rango: se usa dentro de un bucle sobre decenas de miles de pares y una
    comprobación por par costaría más que el cálculo. `zonas_de` valida una vez.
    """
    # A: dentro del 20 %, o las dos en la zona hipoglucémica —donde el 20 % es tan estrecho
    # que no distingue nada y lo que importa es que las dos pidan tratar.
    if abs(predicho - real) <= TOLERANCIA_A * real or (real <= HIPO and predicho <= HIPO):
        return "A"

    # E: tratamiento contrario. Es la peor y va antes que C y D porque los dos cuadrantes
    # opuestos caen también dentro de las condiciones de aquellas.
    if (real >= HIPER and predicho <= HIPO) or (real <= HIPO and predicho >= HIPER):
        return "E"

    # C: sobrecorregir. Arriba, predecir 110 mg/dL de más sobre un valor que no era bajo;
    # abajo, la cuña que empuja a tratar una glucosa que estaba en rango.
    if (HIPO <= real <= 290 and predicho >= real + 110) or (
        130 <= real <= HIPER and predicho <= (7 / 5) * real - 182
    ):
        return "C"

    # D: no detectar. El valor real estaba fuera de rango y la predicción cae dentro, así que
    # nadie hace nada. El 175/3 (≈ 58.3) es donde la frontera de la cuña inferior corta el
    # umbral de 70: por debajo de ese valor cualquier predicción en rango es un fallo de
    # detección, y entre 58.3 y 70 lo es solo si además se desvía más del 20 %.
    if (
        (real >= 240 and HIPO <= predicho <= HIPER)
        or (real <= 175 / 3 and HIPO <= predicho <= HIPER)
        or (175 / 3 <= real <= HIPO and predicho >= (1 + TOLERANCIA_A) * real)
    ):
        return "D"

    return "B"


def zonas_de(reales: Sequence[float], predichos: Sequence[float]) -> list[str]:
    """La zona de cada par. `strict=True` a propósito: dos listas de distinto largo son un
    error de alineación entre reales y predichos, y alinear mal desplaza todas las métricas
    sin que nada falle.
    """
    if not reales:
        raise ValueError("no hay pares que clasificar")
    return [zona(r, p) for r, p in zip(reales, predichos, strict=True)]


def reparto(reales: Sequence[float], predichos: Sequence[float]) -> dict[str, float]:
    """El por ciento en cada una de las cinco zonas, con las cinco llaves siempre presentes.

    Las cinco aunque alguna sea cero: una tabla del reporte a la que le faltan columnas según
    la corrida no se puede comparar con la de la corrida anterior.
    """
    clasificadas = zonas_de(reales, predichos)
    total = len(clasificadas)
    return {z: 100 * clasificadas.count(z) / total for z in ZONAS}


def porcentaje_ab(reales: Sequence[float], predichos: Sequence[float]) -> float:
    """El criterio principal del proyecto: por ciento en A+B. Meta ≥ 95 %."""
    clasificadas = zonas_de(reales, predichos)
    return 100 * sum(1 for z in clasificadas if z in ("A", "B")) / len(clasificadas)
