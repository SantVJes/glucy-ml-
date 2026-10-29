"""Cómo se reparten los datos. **Es lo que más fácilmente invalida un resultado entero.**

Dos reglas, y las dos se rompen solas si nadie las escribe:

1. **Por persona, nunca al azar.** Repartir al azar filas de la misma persona deja que el
   modelo la memorice: aprende «esta fisiología a esta hora suele estar en 140» en lugar de
   aprender a predecir, y la métrica sale espectacular y no significa nada. Es el error más
   común en predicción de glucosa y el que invalida un resultado ante cualquiera que sepa
   leerlo.
2. **Dentro de cada persona, el pasado entrena y el futuro prueba.** Usar datos posteriores
   para predecir anteriores es fuga de información y **no existe en producción**: a las 14:00
   el teléfono no tiene las lecturas de las 15:00.

Las dos juntas dicen que `train_test_split(shuffle=True)` de scikit-learn, que es lo primero
que uno escribe, está mal por los dos motivos a la vez. De ahí que este archivo exista en
lugar de una línea en el script de entrenamiento.

La validación cruzada es **dejando un sujeto fuera** (LOSO): con doce sujetos es lo que mejor
aprovecha el conjunto, y cada pliegue responde la pregunta que importa, que es cómo le va al
modelo con una persona que nunca vio.

Y los hiperparámetros se buscan **solo contra validación**. Mirar la prueba para elegirlos
equivale a entrenar con la prueba; el número que sale de ahí no se puede reportar.
"""

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Protocol


class Fila(Protocol):
    """Lo mínimo que una fila tiene que saber decir para poder repartirla.

    Un `Protocol` y no una clase: las filas vienen de `src/ingest` con muchos más campos, y
    este módulo solo necesita de quién son y de cuándo. Pedir una clase concreta obligaría a
    convertir el conjunto entero para repartirlo.
    """

    @property
    def sujeto(self) -> str: ...

    @property
    def orden(self) -> float:
        """Cualquier cosa creciente en el tiempo: un `timestamp`, un índice de muestra."""


@dataclass(frozen=True)
class Reparto:
    """Un reparto concreto, con los sujetos de cada lado a la vista.

    Lleva los sujetos y no solo las filas porque es lo que hay que poder comprobar y lo que hay
    que poder reportar: «entrenado con estos nueve, probado con estos tres» es una frase que
    va en el documento, y una lista de 200 000 filas no lo dice.
    """

    entrenamiento: list
    prueba: list
    sujetos_entrenamiento: tuple[str, ...]
    sujetos_prueba: tuple[str, ...]

    def __post_init__(self) -> None:
        comunes = set(self.sujetos_entrenamiento) & set(self.sujetos_prueba)
        if comunes:
            raise ValueError(
                f"los sujetos {sorted(comunes)} están en los dos lados: el modelo los "
                "memorizaría y la métrica no valdría nada"
            )


def sujetos_de[F: Fila](filas: Sequence[F]) -> tuple[str, ...]:
    """Los sujetos presentes, en orden estable. Ordenados y no en orden de aparición: un
    reparto que cambia según cómo se leyeron los archivos no se puede repetir, y una corrida
    que no se puede repetir no sirve para un trabajo que se va a evaluar.
    """
    return tuple(sorted({fila.sujeto for fila in filas}))


def partir_por_sujeto[F: Fila](filas: Sequence[F], sujetos_de_prueba: Sequence[str]) -> Reparto:
    """Reparte por persona: los sujetos nombrados van enteros a prueba y el resto a
    entrenamiento.

    Levanta `ValueError` si un sujeto de prueba no está en los datos, en lugar de devolver un
    conjunto de prueba vacío o más pequeño de lo pedido. Un nombre mal escrito en la lista de
    prueba daría un reparto silenciosamente distinto del que dice el documento.
    """
    presentes = set(sujetos_de(filas))
    pedidos = list(dict.fromkeys(sujetos_de_prueba))
    if not pedidos:
        raise ValueError("hay que nombrar al menos un sujeto de prueba")
    faltantes = [s for s in pedidos if s not in presentes]
    if faltantes:
        raise ValueError(f"estos sujetos de prueba no están en los datos: {faltantes}")
    if set(pedidos) == presentes:
        raise ValueError("no queda ningún sujeto para entrenar")

    en_prueba = set(pedidos)
    return Reparto(
        entrenamiento=[f for f in filas if f.sujeto not in en_prueba],
        prueba=[f for f in filas if f.sujeto in en_prueba],
        sujetos_entrenamiento=tuple(s for s in sujetos_de(filas) if s not in en_prueba),
        sujetos_prueba=tuple(pedidos),
    )


def pliegues_loso[F: Fila](filas: Sequence[F]) -> list[Reparto]:
    """Validación cruzada dejando un sujeto fuera: un pliegue por persona.

    Con doce sujetos son doce pliegues, cada uno entrenando con once y probando con el que
    quedó fuera. Devuelve la lista completa y no un generador: los pliegues se recorren varias
    veces —una por modelo y una por línea base— y un generador agotado daría cero pliegues en
    la segunda pasada sin que nada falle.
    """
    todos = sujetos_de(filas)
    if len(todos) < 2:
        raise ValueError(f"hacen falta al menos dos sujetos para LOSO, hay {len(todos)}")
    return [partir_por_sujeto(filas, [sujeto]) for sujeto in todos]


def partir_en_el_tiempo[F: Fila](
    filas: Sequence[F], fraccion_entrenamiento: float = 0.8
) -> dict[str, list[F]]:
    """Dentro de cada persona, el pasado entrena y el futuro prueba.

    Se corta **por sujeto y no sobre el conjunto entero**: los sujetos no comparten calendario
    —cada uno aportó sus ocho semanas cuando le tocó—, así que un corte por fecha global
    dejaría a unos enteros en entrenamiento y a otros enteros en prueba, que es otra cosa.

    Devuelve un diccionario con las dos listas, cada una ordenada en el tiempo. No se baraja
    en ningún momento.
    """
    if not 0 < fraccion_entrenamiento < 1:
        raise ValueError(f"la fracción tiene que estar entre 0 y 1, no {fraccion_entrenamiento}")

    entrenamiento: list[F] = []
    prueba: list[F] = []
    for sujeto in sujetos_de(filas):
        suyas = sorted((f for f in filas if f.sujeto == sujeto), key=lambda f: f.orden)
        corte = int(len(suyas) * fraccion_entrenamiento)
        # Con pocas filas el corte puede dejar un lado vacío. Se avisa en lugar de devolver un
        # conjunto de prueba de cero filas, que daría métricas sobre nada.
        if corte == 0 or corte == len(suyas):
            raise ValueError(
                f"el sujeto {sujeto!r} tiene {len(suyas)} filas: con una fracción de "
                f"{fraccion_entrenamiento} un lado queda vacío"
            )
        entrenamiento.extend(suyas[:corte])
        prueba.extend(suyas[corte:])
    return {"entrenamiento": entrenamiento, "prueba": prueba}


def hay_fuga_temporal[F: Fila](entrenamiento: Sequence[F], prueba: Sequence[F]) -> list[str]:
    """Los sujetos en los que alguna fila de entrenamiento es posterior a una de prueba.

    Es la comprobación que se corre **después** de repartir, no en lugar de repartir bien: una
    aserción sobre el resultado, porque la fuga temporal no se ve mirando el código que la
    causó. Devuelve la lista de sujetos afectados para que el mensaje diga en quién.
    """
    ultima_de_entrenamiento: dict[Hashable, float] = {}
    primera_de_prueba: dict[Hashable, float] = {}
    for fila in entrenamiento:
        anterior = ultima_de_entrenamiento.get(fila.sujeto)
        if anterior is None or fila.orden > anterior:
            ultima_de_entrenamiento[fila.sujeto] = fila.orden
    for fila in prueba:
        anterior = primera_de_prueba.get(fila.sujeto)
        if anterior is None or fila.orden < anterior:
            primera_de_prueba[fila.sujeto] = fila.orden

    return sorted(
        str(sujeto)
        for sujeto, ultima in ultima_de_entrenamiento.items()
        if sujeto in primera_de_prueba and ultima >= primera_de_prueba[sujeto]
    )
