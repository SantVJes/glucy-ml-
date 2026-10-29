"""El artefacto de un modelo y su manifiesto. **Un resultado que no se puede repetir no sirve.**

Ese es el motivo de que este archivo exista en lugar de un `model.save()` suelto en cada
guion de entrenamiento. Toda corrida deja, junto al modelo:

- **la semilla**, los **hiperparámetros** y la **versión**, que es lo que permite repetirla;
- **la lista de características en su orden**, que es lo que `GET /predict` del backend compara
  antes de cargar el artefacto; sin ella, una columna desplazada da números plausibles y
  equivocados sin que nada falle;
- **las métricas**, con el tamaño de la muestra y los sujetos de cada lado, que es lo que va al
  reporte y a la tabla `modelos` del backend (RF-25);
- **de dónde salieron los datos**. Un artefacto entrenado con las series de juguete de
  `src/datos/sintetico.py` queda marcado como `sintetico`, y sus métricas **no se pueden
  reportar**: la señal la generó una suma de senos y cualquier modelo decente sale excelente.

El manifiesto es JSON a propósito: lo lee el backend, lo lee el cuaderno y lo lee una persona.
"""

import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.eval.metrics import Resultado

# Los orígenes posibles de los datos de una corrida. `sintetico` y `simulador` existen para que
# una métrica que no se puede reportar **lo diga en el archivo** y no solo en la memoria de
# quien la corrió.
ORIGENES = ("ohiot1dm", "simulador", "sintetico")

VERSION_DEL_MANIFIESTO = 1


@dataclass(frozen=True)
class Manifiesto:
    """Todo lo que hay que saber de una corrida sin tener que volver a correrla.

    `caracteristicas` va en orden y es el contrato con el pipeline del backend.
    `sujetos_entrenamiento` y `sujetos_prueba` van completos porque «entrenado con estos once,
    probado con este» es una frase del documento, y porque son lo que permite comprobar después
    que nadie se coló a los dos lados.

    `protocolo` dice en una línea cómo se obtuvieron las métricas, y **hace falta** porque el
    caso normal del proyecto es confuso si no se explica: las métricas salen de la validación
    cruzada dejando un sujeto fuera, y el artefacto que se publica se entrena con **todos** los
    sujetos. Sin esa línea, `sujetos_prueba` vacío se lee como «no se evaluó».
    """

    modelo: str
    version: str
    algoritmo: str
    horizonte_min: int
    juego: str
    caracteristicas: list[str]
    semilla: int
    hiperparametros: dict[str, Any]
    origen_datos: str
    n_entrenamiento: int
    n_prueba: int
    sujetos_entrenamiento: list[str]
    sujetos_prueba: list[str]
    metricas: dict[str, Any]
    protocolo: str = ""
    metricas_por_sujeto: dict[str, Any] = field(default_factory=dict)
    metricas_lineas_base: dict[str, Any] = field(default_factory=dict)
    creado_utc: str = ""
    entorno: dict[str, str] = field(default_factory=dict)
    version_del_manifiesto: int = VERSION_DEL_MANIFIESTO

    def __post_init__(self) -> None:
        if self.origen_datos not in ORIGENES:
            raise ValueError(
                f"origen de datos desconocido: {self.origen_datos!r}. Los válidos son {ORIGENES}"
            )
        if not self.caracteristicas:
            raise ValueError(
                "el manifiesto tiene que llevar la lista de características en su orden: es lo "
                "que GET /predict compara antes de cargar el artefacto"
            )
        comunes = set(self.sujetos_entrenamiento) & set(self.sujetos_prueba)
        if comunes:
            raise ValueError(
                f"los sujetos {sorted(comunes)} están en los dos lados: la métrica no valdría "
                "nada y el manifiesto no debe poder decir lo contrario"
            )

    @property
    def se_puede_reportar(self) -> bool:
        """Si las métricas de esta corrida pueden ir al documento final.

        Solo con datos reales. Con el simulador fisiológico se pueden reportar **diciendo que
        son del simulador**, que es otra decisión y la toma quien escribe el reporte; con las
        series de juguete, nunca.
        """
        return self.origen_datos == "ohiot1dm"

    def a_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False, sort_keys=False)


def entorno_actual() -> dict[str, str]:
    """Qué intérprete y qué sistema corrieron el entrenamiento.

    No es burocracia: una diferencia de versión de la biblioteca que entrena cambia los
    números, y cuando dos corridas no coinciden es lo primero que hay que descartar.
    """
    datos = {
        "python": sys.version.split()[0],
        "sistema": f"{platform.system()} {platform.machine()}",
    }
    for nombre, modulo in (("xgboost", "xgboost"), ("scikit-learn", "sklearn")):
        try:
            datos[nombre] = __import__(modulo).__version__
        except ImportError:
            continue
    return datos


def manifiesto_de(
    *,
    modelo: str,
    version: str,
    algoritmo: str,
    horizonte_min: int,
    juego: str,
    caracteristicas: list[str],
    semilla: int,
    hiperparametros: dict[str, Any],
    origen_datos: str,
    n_entrenamiento: int,
    resultado: Resultado,
    sujetos_entrenamiento: list[str],
    sujetos_prueba: list[str],
    protocolo: str = "",
    lineas_base: dict[str, Resultado] | None = None,
    por_sujeto: dict[str, Resultado] | None = None,
) -> Manifiesto:
    """Arma el manifiesto a partir del `Resultado` de la evaluación.

    Guarda **también las métricas de las líneas base de la misma partición**, y no en un archivo
    aparte: «RMSE 18» no significa nada sin el «la persistencia dio 22» al lado, y separarlos
    es cómo se acaba comparando contra un número de otra corrida.
    """
    return Manifiesto(
        modelo=modelo,
        version=version,
        algoritmo=algoritmo,
        horizonte_min=horizonte_min,
        juego=juego,
        caracteristicas=list(caracteristicas),
        semilla=semilla,
        hiperparametros=dict(hiperparametros),
        origen_datos=origen_datos,
        n_entrenamiento=n_entrenamiento,
        n_prueba=resultado.n,
        sujetos_entrenamiento=list(sujetos_entrenamiento),
        sujetos_prueba=list(sujetos_prueba),
        metricas=asdict(resultado),
        protocolo=protocolo,
        metricas_por_sujeto={n: asdict(r) for n, r in (por_sujeto or {}).items()},
        metricas_lineas_base={nombre: asdict(r) for nombre, r in (lineas_base or {}).items()},
        creado_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        entorno=entorno_actual(),
    )


def ruta_del_artefacto(carpeta: Path | str, modelo: str, version: str) -> Path:
    """`artifacts/m1-v0.1.0.json` y su gemelo `.ubj` para el modelo.

    El nombre lleva la versión porque **los artefactos no se sobrescriben**: la tabla
    `modelos` del backend y cada predicción guardan con qué versión se calcularon (RF-25), y
    sobrescribir `m1.json` rompería ese rastro sin avisar.
    """
    return Path(carpeta) / f"{modelo}-{version}"


def guardar(
    manifiesto: Manifiesto,
    guardar_modelo,
    carpeta: Path | str = "artifacts",
) -> dict[str, Path]:
    """Escribe el manifiesto y pide al llamador que escriba el modelo junto a él.

    `guardar_modelo` recibe la ruta base y guarda el modelo con la extensión que le
    corresponda. Se recibe como función en lugar de importar xgboost aquí para que este módulo
    y sus pruebas **no dependan de xgboost**: la integración continua instala solo pytest y
    ruff, y la lógica del manifiesto es justo lo que conviene tener probado.

    **No sobrescribe.** Un artefacto con la misma versión que otro es casi siempre un descuido,
    y perder el que ya estaba se descubre cuando ya no se puede recuperar.
    """
    base = ruta_del_artefacto(carpeta, manifiesto.modelo, manifiesto.version)
    base.parent.mkdir(parents=True, exist_ok=True)
    # `base.with_suffix(".json")` estaría mal: la versión trae puntos, así que `m1-v0.1.0` tiene
    # un «sufijo» `.0` que `with_suffix` reemplazaría, y el manifiesto saldría como
    # `m1-v0.1.json`, con un nombre distinto del `.ubj` del modelo y pisando el de otra versión
    # de parche. Se concatena.
    ruta_json = Path(f"{base}.json")
    if ruta_json.exists():
        raise FileExistsError(
            f"ya existe {ruta_json}: los artefactos no se sobrescriben, hay que subir la "
            "versión. Si de verdad sobra, bórralo a mano."
        )
    ruta_json.write_text(manifiesto.a_json(), encoding="utf-8")
    rutas = {"manifiesto": ruta_json}
    ruta_modelo = guardar_modelo(base)
    if ruta_modelo is not None:
        rutas["modelo"] = Path(ruta_modelo)
    return rutas


def leer_manifiesto(ruta: Path | str) -> Manifiesto:
    """Vuelve a cargar un manifiesto, con sus comprobaciones puestas.

    Pasa por `Manifiesto` y no por `json.load` a secas para que un archivo editado a mano con
    un sujeto en los dos lados, o sin lista de características, falle al leerse y no al usarse.
    """
    datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
    conocidos = {campo for campo in Manifiesto.__dataclass_fields__}
    desconocidos = set(datos) - conocidos
    if desconocidos:
        raise ValueError(
            f"el manifiesto {ruta} trae campos que este código no conoce: "
            f"{sorted(desconocidos)}. Probablemente lo escribió una versión más nueva."
        )
    return Manifiesto(**datos)
