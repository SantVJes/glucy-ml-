"""El bosque que lleva dentro un `.mlmodel`, leído y evaluado en Python puro.

Existe por un motivo concreto: **la conversión a Core ML se puede comprobar sin una Mac.**
`coremltools` convierte en Linux pero no puede *ejecutar* el modelo —`libcoremlpython` es solo
de macOS—, así que la comprobación obvia, «predice lo mismo que XGBoost», no se puede hacer
donde se convierte. Y es justo la comprobación que no se puede saltar: un árbol traducido al
revés da números plausibles y equivocados, el modelo no falla, la app no falla, y la cifra
equivocada es una predicción de glucosa.

Así que en lugar de ejecutar el modelo se **lee su especificación** —que es un protobuf con los
nodos adentro— y se recorre aquí. Si el recorrido de aquí da lo mismo que XGBoost, la
traducción es fiel; lo único que queda fuera es si el sistema del teléfono recorre el árbol
igual, y eso se comprueba una vez en el dispositivo.

Después de `ensamble_de`, este módulo no toca protobuf ni nada externo: son tuplas y aritmética
de la biblioteca estándar.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

# `TreeEnsembleParameters.TreeNode.TreeNodeBehavior`, leído del protobuf de coremltools y no de
# memoria. El orden importa y **no es el intuitivo**: los «menor que» van antes que los «mayor
# que», así que confundirlos invierte cada corte del árbol. Es el error que esta comprobación
# encontró primero, y estaba en el comprobador, no en la conversión.
MENOR_O_IGUAL = 0
MENOR = 1
MAYOR_O_IGUAL = 2
MAYOR = 3
IGUAL = 4
DISTINTO = 5
HOJA = 6


class Nodo(NamedTuple):
    """Un nodo del bosque, ya sin protobuf.

    `faltante_al_verdadero` es la dirección que XGBoost aprendió para los valores que faltan.
    No es un detalle: la regla del proyecto es que **lo que falta es `NaN` y nunca cero**, y en
    modo sensor con huecos en la serie hay filas con `NaN` de verdad.
    """

    comportamiento: int
    caracteristica: int
    umbral: float
    hijo_verdadero: int
    hijo_falso: int
    faltante_al_verdadero: bool
    valor: float


@dataclass(frozen=True)
class Ensamble:
    """El bosque completo: los nodos por (árbol, nodo), las raíces y el valor de base."""

    nodos: dict[tuple[int, int], Nodo]
    arboles: tuple[int, ...]
    base: float

    @property
    def cortes(self) -> int:
        return sum(1 for n in self.nodos.values() if n.comportamiento != HOJA)

    @property
    def cortes_con_faltante_al_verdadero(self) -> int:
        """Cuántos cortes mandan el valor que falta al hijo verdadero.

        Es el número que decide si el `NaN` importa en este modelo: si es cero, un sistema que
        no entienda la marca de faltante igual acierta, porque una comparación IEEE con `NaN`
        da falso y el hijo falso es el que tocaba.
        """
        return sum(
            1 for n in self.nodos.values() if n.comportamiento != HOJA and n.faltante_al_verdadero
        )

    def predecir(self, fila: Sequence[float], *, respetar_faltante: bool = True) -> float:
        """Recorre los árboles y suma. Con `respetar_faltante=False` imita a un sistema que
        compare el `NaN` como cualquier número, que es lo que hace la aritmética IEEE a secas."""
        total = self.base
        for arbol in self.arboles:
            nodo = self.nodos[(arbol, 0)]
            while nodo.comportamiento != HOJA:
                valor = fila[nodo.caracteristica]
                falta = valor is None or math.isnan(valor)
                if falta and respetar_faltante:
                    va_al_verdadero = nodo.faltante_al_verdadero
                else:
                    va_al_verdadero = _compara(nodo.comportamiento, valor, nodo.umbral)
                siguiente = nodo.hijo_verdadero if va_al_verdadero else nodo.hijo_falso
                nodo = self.nodos[(arbol, siguiente)]
            total += nodo.valor
        return total

    def predecir_muchas(
        self, filas: Iterable[Sequence[float]], *, respetar_faltante: bool = True
    ) -> list[float]:
        return [self.predecir(f, respetar_faltante=respetar_faltante) for f in filas]


def _compara(comportamiento: int, valor: float, umbral: float) -> bool:
    if comportamiento == MENOR_O_IGUAL:
        return valor <= umbral
    if comportamiento == MENOR:
        return valor < umbral
    if comportamiento == MAYOR_O_IGUAL:
        return valor >= umbral
    if comportamiento == MAYOR:
        return valor > umbral
    if comportamiento == IGUAL:
        return valor == umbral
    if comportamiento == DISTINTO:
        return valor != umbral
    raise ValueError(f"comportamiento de nodo desconocido: {comportamiento}")


def ensamble_de(spec) -> Ensamble:
    """Saca el bosque de la especificación de un `.mlmodel` de regresión por árboles.

    Revienta si el spec no es un `treeEnsembleRegressor`: convertir un clasificador por
    descuido y comprobarlo como regresión daría un número sin sentido en lugar de un error.
    """
    if not spec.HasField("treeEnsembleRegressor"):
        tipo = spec.WhichOneof("Type")
        raise ValueError(
            f"la especificación no es un treeEnsembleRegressor sino un {tipo!r}: este "
            "comprobador solo sabe recorrer el bosque de un modelo de regresión"
        )
    bosque = spec.treeEnsembleRegressor.treeEnsemble
    nodos: dict[tuple[int, int], Nodo] = {}
    for n in bosque.nodes:
        # Una hoja puede tener varias dimensiones de salida; aquí la regresión es de una, y
        # sumarlas es lo correcto para esa única dimensión.
        valor = sum(v.evaluationValue for v in n.evaluationInfo)
        nodos[(n.treeId, n.nodeId)] = Nodo(
            comportamiento=n.nodeBehavior,
            caracteristica=n.branchFeatureIndex,
            umbral=n.branchFeatureValue,
            hijo_verdadero=n.trueChildNodeId,
            hijo_falso=n.falseChildNodeId,
            faltante_al_verdadero=bool(n.missingValueTracksTrueChild),
            valor=valor,
        )
    base = list(bosque.basePredictionValue) or [0.0]
    if len(base) != 1:
        raise ValueError(
            f"el modelo predice {len(base)} dimensiones y aquí se espera una sola: "
            "los tres modelos de Glucy predicen un número"
        )
    return Ensamble(
        nodos=nodos,
        arboles=tuple(sorted({n.treeId for n in bosque.nodes})),
        base=base[0],
    )
