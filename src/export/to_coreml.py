"""Convierte el artefacto de XGBoost a Core ML, **y comprueba que predice lo mismo**.

Es el último tramo del camino del modelo: entrenamiento aquí → artefacto convertido a Core ML →
se distribuye con la app → **la inferencia corre dentro del teléfono**, sin red, y ningún dato
de glucosa sale de ahí para predecir.

La conversión en sí son tres líneas de `coremltools`. Lo que ocupa este archivo es lo otro: que
el modelo convertido dé los mismos números que el que se midió. Sin esa comprobación, el
proyecto reporta las métricas de un modelo y el teléfono ejecuta otro, y nadie se enteraría:
un árbol traducido al revés no falla, predice mal. Y como `coremltools` avisa de que la última
versión de XGBoost que probó es la 1.4.2, dar por bueno lo que salga sería confiar en una
compatibilidad que la propia herramienta no afirma.

```bash
python -m src.export.to_coreml --artefacto artifacts/m1-v0.1.0.json --datos sintetico
```
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from src.export.arbol import ensamble_de
from src.train.artefacto import entorno_actual, leer_manifiesto

# Cuánto se admite de diferencia entre XGBoost y el modelo convertido, en mg/dL. No es una
# tolerancia clínica: es el error de acumular en 32 bits cientos de hojas, y se mide en
# diezmilésimas. Un milésimo de mg/dL ya es mil veces más de lo que hace falta, así que si se
# pasa de aquí, lo que cambió es la traducción y no el redondeo.
TOLERANCIA_MG_DL = 1e-3

# A partir de cuánto una diferencia por valores que faltan deja de ser un detalle numérico y
# pasa a ser otra predicción. Cinco mg/dL es menos de lo que separa dos lecturas consecutivas
# de un sensor, y aun así ya mueve una alerta cerca del umbral.
DIFERENCIA_QUE_IMPORTA_MG_DL = 5.0

AVISO_MEDICO = (
    "Glucy no es un dispositivo medico certificado. Informa; no prescribe ni administra, y "
    "jamas calcula ni sugiere una dosis de insulina."
)


@dataclass(frozen=True)
class Comprobacion:
    """El resultado de comparar el modelo convertido contra el original, fila por fila.

    Dos números y no uno, porque hay dos preguntas distintas:

    - `max_diferencia` responde «¿la traducción es fiel?». Se mide con las entradas
      redondeadas a 32 bits, que es lo que XGBoost hace por dentro, y tiene que ser ruido.
    - `max_diferencia_con_doble_precision` responde «¿y si la app manda `Double`?». Los
      umbrales del árbol son de 32 bits; comparar contra ellos en doble precisión voltea algún
      corte y la predicción se mueve. **Es una instrucción para el código de Swift.**
    - `max_diferencia_ignorando_faltantes` responde «¿y si el sistema del teléfono no entiende
      la marca de valor faltante y compara el `NaN` como cualquier número?». Ese número puede
      ser grande, y entonces hay algo que comprobar en el dispositivo.
    """

    n: int
    n_filas_con_faltantes: int
    max_diferencia: float
    media_diferencia: float
    max_diferencia_con_doble_precision: float
    filas_que_cambian_con_doble_precision: int
    max_diferencia_ignorando_faltantes: float
    filas_que_cambian_mas_de_lo_que_importa: int
    cortes: int
    cortes_con_faltante_al_verdadero: int

    @property
    def la_traduccion_es_fiel(self) -> bool:
        return self.max_diferencia <= TOLERANCIA_MG_DL

    @property
    def hay_que_mandar_float_de_32_bits(self) -> bool:
        """Si la precisión de la entrada cambia la predicción en este modelo."""
        return self.max_diferencia_con_doble_precision > TOLERANCIA_MG_DL

    @property
    def el_faltante_cambia_la_prediccion(self) -> bool:
        """Si la dirección del valor que falta puede cambiar la predicción en este modelo.

        Cuando es falso, no hay nada que comprobar en el teléfono: o no hay filas con valores
        que falten, o ningún corte manda el faltante al hijo verdadero, y entonces una
        comparación IEEE a secas acierta por su cuenta.
        """
        return self.max_diferencia_ignorando_faltantes > DIFERENCIA_QUE_IMPORTA_MG_DL

    def a_texto(self) -> str:
        renglones = [
            f"filas comprobadas: {self.n} ({self.n_filas_con_faltantes} con valores que faltan)",
            f"cortes del bosque: {self.cortes}, de los cuales "
            f"{self.cortes_con_faltante_al_verdadero} mandan el faltante al hijo verdadero",
            f"diferencia máxima contra XGBoost: {self.max_diferencia:.3g} mg/dL "
            f"(media {self.media_diferencia:.3g})",
            f"{'FIEL' if self.la_traduccion_es_fiel else 'NO ES FIEL'}: la tolerancia es "
            f"{TOLERANCIA_MG_DL:g} mg/dL",
        ]
        if self.hay_que_mandar_float_de_32_bits:
            renglones += [
                "",
                "La app tiene que mandar cada característica como Float (32 bits) y no como "
                f"Double: con doble precisión la predicción se desvía hasta "
                f"{self.max_diferencia_con_doble_precision:.3g} mg/dL en "
                f"{self.filas_que_cambian_con_doble_precision} de {self.n} filas, porque los "
                "umbrales del árbol son de 32 bits y XGBoost compara contra ellos con esa "
                "misma precisión.",
            ]
        if self.el_faltante_cambia_la_prediccion:
            renglones += [
                "",
                "AVISO, y hay que comprobarlo en el iPhone: si el sistema del teléfono no "
                "respeta la marca de valor faltante y compara el NaN como cualquier número, "
                f"la predicción se desvía hasta {self.max_diferencia_ignorando_faltantes:.3g} "
                f"mg/dL, y {self.filas_que_cambian_mas_de_lo_que_importa} de {self.n} filas "
                f"cambian más de {DIFERENCIA_QUE_IMPORTA_MG_DL:g} mg/dL.",
            ]
        return "\n".join(renglones)


def convertir(modelo, caracteristicas: Sequence[str], manifiesto=None):
    """El artefacto de XGBoost como `MLModel`, con sus metadatos puestos.

    Los metadatos no son adorno. La **lista de características en su orden** viaja dentro del
    archivo porque es el contrato con el código de Swift que lo llama: una columna desplazada
    da números plausibles y equivocados sin que nada falle, y con la lista dentro del modelo el
    teléfono puede comprobarlo. Y va el aviso de que Glucy no es un dispositivo médico, porque
    el archivo se distribuye y se puede abrir solo.
    """
    import coremltools as ct

    convertido = ct.converters.xgboost.convert(
        modelo,
        feature_names=list(caracteristicas),
        mode="regressor",
        # Los umbrales del modelo son float32 y en Swift se leen como Double. Sin esto, la
        # conversión sube los umbrales a float64 y un valor justo en el umbral puede caer del
        # otro lado: la diferencia es minúscula y aparece solo en algunas filas, que es la
        # clase de error que no se encuentra después.
        force_32bit_float=True,
    )
    convertido = ct.models.MLModel(_con_el_valor_de_base_corregido(convertido, modelo))
    convertido.author = "Glucy"
    convertido.license = "MIT"
    convertido.short_description = (
        f"Glucosa predicha en mg/dL. {AVISO_MEDICO}"
        if manifiesto is None
        else (
            f"Glucosa predicha en mg/dL a {manifiesto.horizonte_min} minutos "
            f"({manifiesto.modelo} {manifiesto.version}, juego {manifiesto.juego}). "
            f"{AVISO_MEDICO}"
        )
    )
    metadatos = {
        "glucy.caracteristicas": ",".join(caracteristicas),
        "glucy.unidad": "mg/dL",
        "glucy.aviso": AVISO_MEDICO,
    }
    if manifiesto is not None:
        convertido.version = manifiesto.version
        metadatos |= {
            "glucy.modelo": manifiesto.modelo,
            "glucy.version": manifiesto.version,
            "glucy.horizonte_min": str(manifiesto.horizonte_min),
            "glucy.juego": manifiesto.juego,
            "glucy.origen_datos": manifiesto.origen_datos,
            # Que el archivo diga si sus métricas se pueden reportar. Es el mismo criterio del
            # manifiesto (RF-25) y la protección contra un artefacto de corrida sintética
            # llegando al teléfono o al documento sin que nadie lo note.
            "glucy.se_puede_reportar": str(manifiesto.se_puede_reportar).lower(),
        }
    convertido.user_defined_metadata.update(metadatos)
    return convertido


def base_de(modelo) -> float:
    """El `base_score` que el modelo aprendió de los datos.

    Se lee de la configuración del booster y no de `intercept_` porque así funciona también con
    un `Booster` a secas, sin el envoltorio de scikit-learn.
    """
    import json as _json

    config = _json.loads(modelo.get_booster().save_config())
    valor = config["learner"]["learner_model_param"]["base_score"]
    # Viene como texto y a veces entre corchetes, p. ej. "[1.4372429E2]".
    return float(str(valor).strip("[]"))


def _con_el_valor_de_base_corregido(convertido, modelo):
    """Devuelve el spec con el `base_score` de verdad. **Sin esto, la conversión está mal.**

    Es el error más grave que encontró la comprobación, y no avisa de ninguna forma:

    - Desde XGBoost 2.0, si no se fija `base_score` el modelo **lo aprende de los datos**. Con
      glucosa, eso es la media de la serie: en una corrida de prueba, 143.72 mg/dL.
    - `coremltools` escribe en el spec un `basePredictionValue` de **0.5**, que era el valor por
      omisión de XGBoost 1.x, en lugar de leer el que el modelo trae.

    El resultado es que el modelo del teléfono predeciría **143 mg/dL por debajo** del que se
    midió, siempre: una glucosa real de 140 saldría como −3. No falla nada: el archivo se
    convierte, se guarda, se carga en el teléfono y predice. La propia herramienta avisa de que
    la última versión de XGBoost con la que se probó es la 1.4.2, y este es exactamente el tipo
    de cosa que cambió desde entonces.

    Devuelve el spec en lugar de modificar el modelo porque **`get_spec()` entrega una copia**:
    cambiarla no cambia el modelo, y la corrección se perdería sin que nada avisara. El
    llamador arma un `MLModel` nuevo con el spec corregido.
    """
    spec = convertido.get_spec()
    bosque = spec.treeEnsembleRegressor.treeEnsemble
    del bosque.basePredictionValue[:]
    bosque.basePredictionValue.append(base_de(modelo))
    return spec


def a_32_bits(valor: float) -> float:
    """El mismo número redondeado a precisión simple, con la biblioteca estándar.

    Es lo que hace XGBoost con cada entrada antes de comparar, y es lo que la app tiene que
    hacer antes de llamar al modelo.
    """
    return struct.unpack("f", struct.pack("f", valor))[0]


def comprobar(
    spec,
    predecir: Callable[[Sequence[Sequence[float]]], Sequence[float]],
    filas: Sequence[Sequence[float]],
) -> Comprobacion:
    """Compara el bosque del `.mlmodel` contra el modelo original, fila por fila.

    `predecir` se recibe como función para que este módulo no importe xgboost: lo que se
    compara es «lo que predice el original», venga de donde venga.
    """
    if not filas:
        raise ValueError("no hay filas con las que comprobar la conversión")
    ensamble = ensamble_de(spec)
    # XGBoost cuantiza las entradas a 32 bits por dentro, así que la comparación fiel se hace
    # con los mismos bits. Medir con float64 mezcla dos cosas distintas: si la traducción es
    # fiel, y qué precisión usa quien la llama.
    filas32 = [[a_32_bits(v) for v in fila] for fila in filas]
    esperado = list(predecir(filas32))
    obtenido = ensamble.predecir_muchas(filas32)
    con_doble = ensamble.predecir_muchas(filas)
    ignorando = ensamble.predecir_muchas(filas32, respetar_faltante=False)

    diferencias = [abs(a - b) for a, b in zip(esperado, obtenido, strict=True)]
    con_doble_dif = [abs(a - b) for a, b in zip(esperado, con_doble, strict=True)]
    diferencias_ignorando = [abs(a - b) for a, b in zip(esperado, ignorando, strict=True)]
    con_faltantes = sum(1 for f in filas if any(v is None or math.isnan(v) for v in f))
    return Comprobacion(
        n=len(filas),
        n_filas_con_faltantes=con_faltantes,
        # `float()` a la fuerza: `predecir` suele devolver un arreglo de numpy, y un
        # `float32` de numpy no se serializa a JSON. El informe se escribe en JSON.
        max_diferencia=float(max(diferencias)),
        media_diferencia=float(sum(diferencias) / len(diferencias)),
        max_diferencia_con_doble_precision=float(max(con_doble_dif)),
        filas_que_cambian_con_doble_precision=sum(1 for d in con_doble_dif if d > TOLERANCIA_MG_DL),
        max_diferencia_ignorando_faltantes=float(max(diferencias_ignorando)),
        filas_que_cambian_mas_de_lo_que_importa=sum(
            1 for d in diferencias_ignorando if d > DIFERENCIA_QUE_IMPORTA_MG_DL
        ),
        cortes=ensamble.cortes,
        cortes_con_faltante_al_verdadero=ensamble.cortes_con_faltante_al_verdadero,
    )


def exportar(
    ruta_manifiesto: Path | str,
    filas: Sequence[Sequence[float]],
    *,
    origen_de_las_filas: str = "sin decir",
    carpeta: Path | str | None = None,
) -> tuple[Path, Path, Comprobacion]:
    """Convierte el artefacto que describe ese manifiesto, lo comprueba y lo guarda.

    Devuelve la ruta del `.mlmodel`, la del informe de la comprobación y la comprobación.

    **Si la traducción no es fiel, no se escribe nada.** Un `.mlmodel` que no predice lo que se
    midió es peor que no tenerlo: con el archivo en la carpeta, alguien lo mete a la app.
    """
    import xgboost as xgb

    ruta_manifiesto = Path(ruta_manifiesto)
    manifiesto = leer_manifiesto(ruta_manifiesto)
    base = Path(str(ruta_manifiesto)[: -len(".json")])
    ruta_modelo = Path(f"{base}.ubj")
    if not ruta_modelo.exists():
        raise SystemExit(
            f"falta {ruta_modelo}, el modelo que acompaña al manifiesto. Un manifiesto solo no "
            "se puede convertir."
        )
    modelo = xgb.XGBRegressor()
    modelo.load_model(str(ruta_modelo))

    convertido = convertir(modelo, manifiesto.caracteristicas, manifiesto)
    comprobacion = comprobar(convertido.get_spec(), modelo.predict, filas)
    if not comprobacion.la_traduccion_es_fiel:
        raise SystemExit(
            "la conversión NO es fiel y no se escribió ningún archivo:\n" + comprobacion.a_texto()
        )

    destino = Path(carpeta) if carpeta is not None else base.parent
    destino.mkdir(parents=True, exist_ok=True)
    nombre = f"{manifiesto.modelo}-{manifiesto.version}"
    ruta_mlmodel = destino / f"{nombre}.mlmodel"
    ruta_informe = destino / f"{nombre}-coreml.json"
    # No se sobrescribe, por lo mismo que los artefactos (RF-25): cada predicción guarda con
    # qué versión se calculó, y pisar un .mlmodel rompe ese rastro sin avisar.
    for ruta in (ruta_mlmodel, ruta_informe):
        if ruta.exists():
            raise SystemExit(
                f"ya existe {ruta}: no se sobrescribe, hay que subir la versión del artefacto."
            )
    # El informe se arma **antes** de escribir nada: si algo falla al serializarlo, no queda
    # un `.mlmodel` a medias en la carpeta, que además bloquearía el reintento porque no se
    # sobrescribe.
    informe = (
        json.dumps(
            {
                "modelo": manifiesto.modelo,
                "version": manifiesto.version,
                "caracteristicas": manifiesto.caracteristicas,
                # De dónde salieron las filas con las que se comprobó. Importa: una
                # comprobación sobre series sintéticas dice que la traducción es fiel, no que
                # el modelo sirva, y el archivo tiene que poder distinguirlo.
                "origen_de_las_filas_de_comprobacion": origen_de_las_filas,
                "tolerancia_mg_dl": TOLERANCIA_MG_DL,
                "comprobacion": asdict(comprobacion),
                "la_traduccion_es_fiel": comprobacion.la_traduccion_es_fiel,
                "hay_que_comprobar_el_faltante_en_el_iphone": (
                    comprobacion.el_faltante_cambia_la_prediccion
                ),
                "la_app_tiene_que_mandar_float_de_32_bits": (
                    comprobacion.hay_que_mandar_float_de_32_bits
                ),
                "entorno": entorno_actual(),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    convertido.save(str(ruta_mlmodel))
    ruta_informe.write_text(informe, encoding="utf-8")
    return ruta_mlmodel, ruta_informe, comprobacion


def _filas_de_comprobacion(args: argparse.Namespace, manifiesto) -> list[list[float]]:
    """Las filas con las que se comprueba, armadas con el mismo código que entrena.

    Comprobar con números al azar sería más fácil y mucho peor: no tendrían `NaN` donde la
    serie real los tiene, y el caso de los valores que faltan —el único donde esta conversión
    puede desviarse— no se tocaría nunca.
    """
    from src.features.features import a_matriz
    from src.train.comun import filas_de, registros

    filas = filas_de(registros(args), horizonte_min=manifiesto.horizonte_min)
    matriz = a_matriz([f.caracteristicas for f in filas], manifiesto.juego)
    return matriz[: args.filas]


def main() -> None:
    # Dentro de la función, como el resto de lo que viene de `src.train.comun`: ese módulo
    # arrastra xgboost y este tiene que poder importarse sin él (la integración continua no
    # instala las dependencias del entrenamiento).
    from src.train.comun import CARPETA_SIMULADOR

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--artefacto", required=True, help="el .json del manifiesto a convertir")
    p.add_argument(
        "--datos",
        default="ohiot1dm",
        choices=("ohiot1dm", "simulador", "sintetico"),
        help="de dónde salen las filas con las que se comprueba la conversión",
    )
    p.add_argument("--carpeta", default="data/ohiot1dm", help="carpeta de los XML de OhioT1DM")
    p.add_argument(
        "--carpeta-simulador",
        default=CARPETA_SIMULADOR,
        help="carpeta con los JSON que dejó `python -m src.datos.simulador`",
    )
    p.add_argument("--sujetos", type=int, default=3)
    p.add_argument("--dias", type=int, default=7)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument(
        "--filas", type=int, default=2000, help="cuántas filas se comprueban una por una"
    )
    args = p.parse_args()

    manifiesto = leer_manifiesto(args.artefacto)
    filas = _filas_de_comprobacion(args, manifiesto)
    ruta_mlmodel, ruta_informe, comprobacion = exportar(
        args.artefacto, filas, origen_de_las_filas=args.datos
    )
    print(comprobacion.a_texto())
    print()
    print(f"modelo:  {ruta_mlmodel}")
    print(f"informe: {ruta_informe}")


if __name__ == "__main__":
    main()
