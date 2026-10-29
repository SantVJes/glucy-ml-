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

## El cotejo con las implementaciones públicas

`referencia/reglas_clinicas.py` de `glucy-backend` es el árbitro de las reglas clínicas y este
repositorio tiene que dar sus mismos números. Su `docstring` de `zona_clarke` advertía que la
implementación seguía «la formulación habitual» y que **convenía cotejarla antes de reportar
métricas**. Se cotejó el 29/10/2026 y se rehízo el mismo día, par por par sobre los 337 561
pares enteros de 20 a 600 mg/dL, contra las dos implementaciones públicas que se pueden leer:

- el puerto de Python de la implementación de MATLAB File Exchange,
  https://github.com/suetAndTie/ClarkeErrorGrid, leyendo su código fuente, y
- el código SAS del paper 133-2013 del SAS Global Forum,
  https://support.sas.com/resources/papers/proceedings13/133-2013.pdf.

Con el puerto, este archivo **coincide en los 337 561 pares**. Con el SAS difiere en cuatro
familias; tres son `<` contra `<=` en los umbrales exactos 70, 180 y 240, donde aquí ya se toma
la lectura severa. La cuarta es de fondo y cambió este archivo:

| Pares | Con un real… y un predicho… | Decía | Dice el SAS |
| --- | --- | --- | --- |
| 19 900 | real 291–489, predicho ≥ real + 110 | B | **C** |

**El SAS tiene razón, y el motivo vale la pena saberlo.** El puerto acota la zona C con
`real <= 290`, y 290 es el punto donde la recta `y = x + 110` se sale por arriba de una gráfica
de 0 a 400 mg/dL —literalmente el `set_ylim([0, 400])` de su código, que además imprime un
aviso si recibe valores de más de 400—. **No es una frontera clínica: es el borde del dibujo.**
El rango válido de Glucy es 20 a 600 (RF-05), así que ese tope recortaba la zona C justo donde
este proyecto todavía tiene datos. Un real de 300 predicho como 420 invita a sobrecorregir a lo
grande, y salía contado como error benigno.

## Una corrección a lo que decía este archivo

La versión anterior de este comentario afirmaba que el puerto público contestaba **B** para un
real de 59 a 70 con predicho de 71 a 180, por una condición `p >= (7/5)·real + 110`. Leído el
archivo, la condición del puerto es `p >= (6/5)·real`, la misma de aquí, y contesta **D** igual
que este archivo. No había tal discrepancia: la comparación anterior se hizo contra una
transcripción equivocada del puerto.

## Por qué importa

B cuenta para el A+B que el proyecto promete en ≥ 95 %: cada vez que una formulación llama B a
lo que otra llama C o D, **infla justo la cifra que decide si el modelo sirve**. Donde las
formulaciones públicas difieren, aquí se toma **la severa**. Es la decisión conservadora: si al
cotejar contra el artículo original resultara que la indulgente es la correcta, el proyecto
habría reportado una métrica **peor** que la real, que es el error que se puede defender. Al
contrario, no.

`tests/test_clarke.py` deja escritos los casos concretos, así que volver a cualquiera de las
otras formulaciones hace fallar una prueba en lugar de mover la métrica en silencio.

**Lo que sigue pendiente** es cotejar contra el artículo original de Clarke et al. (1987), que
no está accesible en línea sin suscripción. Mientras no se haga, el documento final tiene que
decir **qué formulación se usó**, no solo «rejilla de Clarke».
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
    #
    # Sin tope por arriba, a propósito: el `real <= 290` del puerto público es el punto donde
    # la recta se sale de una gráfica de 400 mg/dL, no una frontera clínica, y aquí el rango
    # llega a 600.
    if (real >= HIPO and predicho >= real + 110) or (
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
