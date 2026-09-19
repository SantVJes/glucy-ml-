# CLAUDE.md — glucy-ml

Este archivo va en la raíz del repositorio `glucy-ml`. Claude Code lo lee solo al abrir el
proyecto, así que todo lo que esté aquí se aplica sin que nadie tenga que repetirlo.

---

## Qué es este repositorio

Entrenamiento, validación y artefactos del modelo predictivo de **Glucy**. Python con XGBoost,
scikit-learn y SHAP. Aquí no hay interfaz ni API: hay datos, características, modelos,
métricas y un cuaderno de evaluación.

Proyecto académico (Universidad Tecmilenio, 8.º semestre) de **una sola persona**, Jesús
Santiago Velasco. Entrega final: **finales de noviembre de 2026**. Este repositorio
corresponde a la **fase 3**, que es el cuello de botella del proyecto: sin datos no hay modelo.

Este repositorio es **uno de tres**:

| Repositorio | Qué contiene | Cómo se conecta con este |
| --- | --- | --- |
| glucy-ios | app SwiftUI | ejecuta con **Core ML** el artefacto que sale de aquí, dentro del teléfono |
| glucy-backend | API y pipeline | carga el artefacto versionado en memoria y registra sus métricas en la tabla `modelos` |
| **glucy-ml** (este) | entrenamiento, validación, evaluación, conversión a Core ML | — |

## Idioma y comentarios (regla del proyecto)

- **Todo en español**: comentarios, docstrings, mensajes de commit, ramas, pull requests,
  issues y el texto de los cuadernos.
- Los nombres de las características se escriben como están especificados y **se mantienen
  idénticos en los tres repositorios**: `cob`, `iob`, `rezago_5`, `rezago_30`, `derivada_1`,
  `hora_sin`, `hora_cos`, `antiguedad_min`, `contexto_comida`. Si aquí se llama distinto que
  en el pipeline del backend, la predicción se alimenta con las columnas cambiadas de lugar y
  nadie lo nota hasta que las métricas salen raras.
- **Docstrings en todo lo que decide algo**, y se documenta el porqué:
  ```python
  def partir_por_sujeto(datos):
      """Reparte por persona, nunca al azar.

      Repartir al azar filas de la misma persona deja que el modelo la memorice e infle la
      métrica. Es el error más común en predicción de glucosa y el que invalida un resultado.
      """
  ```
- Cuando algo viene de un requisito o una decisión, **se cita su identificador** (`RF-26`,
  `RF-24b`, `D-8`).

## Los tres modelos

| Modelo | Predice | Entradas | Modo |
| --- | --- | --- | --- |
| **M-1** | glucosa a 30 min | rezagos, derivadas, medias móviles, COB, IOB, hora cíclica, actividad, sueño | sensor |
| **M-2** | glucosa a 60 min | las mismas | sensor |
| **M-3** | valor máximo después de una comida y cuándo ocurre | última lectura y su antigüedad, contexto, COB, IOB, hora cíclica | sin sensor |

Es **regresión sobre series de tiempo**, no clasificación de «va a haber hipoglucemia o no». Se
predice el valor y **la alerta se deriva de él** comparándolo con el umbral del perfil. Así el
usuario ve la curva y entiende por qué se disparó la alerta.

Un modelo independiente por horizonte (RF-26) y uno aparte para el pico postprandial (RF-27).
**No es un modelo que se degrada**: en modo sin sensor no hay serie continua, y un modelo que
dependa de rezagos entregaría basura con apariencia de predicción.

## El protocolo de validación, que es lo que más se puede arruinar

| Decisión | Cómo queda |
| --- | --- |
| Partición | **Por persona**, nunca al azar. Entrenar con unos sujetos, probar con sujetos que el modelo jamás vio. |
| Orden temporal | Dentro de cada sujeto, entrenar con el pasado y probar con el futuro. **Nunca barajar la serie**: usar datos posteriores para predecir anteriores es fuga de información y no existe en producción. |
| Validación cruzada | Dejando **un sujeto fuera** cada vez. Con doce sujetos es lo que mejor aprovecha el conjunto. |
| Hiperparámetros | Búsqueda evaluada **solo en validación**. Tocarlos mirando la prueba equivale a entrenar con la prueba. |
| Líneas base **obligatorias** | Persistencia («dentro de 30 min será igual que ahora»), extrapolación lineal, Ridge, Random Forest y LSTM. **Un modelo que no le gana a la persistencia no justifica su existencia.** |
| Versionado | Cada modelo se guarda con versión, algoritmo, horizonte, fecha y métricas, y cada predicción registra con qué versión se calculó. |

## Métricas y metas

| Métrica | Meta |
| --- | --- |
| **Rejilla de Clarke, zonas A+B** | **≥ 95 %** — el criterio principal: clasifica el error por su consecuencia clínica, no por su tamaño |
| **Sensibilidad a hipoglucemia** | **≥ 85 %** |
| **Tasa de falsas alarmas** | **≤ 15 %** — criterio de aceptación: un modelo que avisa de más se apaga solo |
| RMSE a 30 min | ≤ 20 mg/dL |
| MAE a 30 min | ≤ 15 mg/dL |
| MARD | ≤ 12 % |
| Error del pico postprandial (M-3) | ≤ 25 mg/dL |
| Tiempo de anticipación | se reporta, con mediana y rango |

Cuidado con una errata conocida: el documento de la fase F3 en Drive trae 80 % y 20 % para
sensibilidad y falsas alarmas. **Los valores buenos son 85 % y 15 %**, y así están en el
reporte final.

## Datos: OhioT1DM

12 personas con diabetes tipo 1, 8 semanas cada una. Glucosa continua cada 5 min, glucosa de
punción, insulina basal y en bolo, comidas con carbohidratos estimados por la propia persona,
ejercicio con intensidad **1–10**, sueño con calidad **1–3**, estrés, enfermedad y datos de
banda de actividad. Gratuito para uso académico, **previa solicitud formal y acuerdo de uso de
datos**.

Reglas duras sobre los datos:

- **El conjunto no se sube al repositorio.** `data/` va en `.gitignore`; lo prohíbe el acuerdo
  de uso. El repositorio es público.
- **Las escalas 1–10 y 1–3 no se cambian**: son las de OhioT1DM, y por eso los campos de la
  app se diseñaron así. Cambiarlas obligaría a traducciones que introducen error.
- **El peso no entra al modelo.** El conjunto no lo trae, y dentro de una misma persona es
  constante en un horizonte de 30 minutos. Inventar una personalización que los datos no
  sostienen sería peor que no tenerla.
- Si el acceso no llega a tiempo, el respaldo es un **simulador fisiológico de código
  abierto**, con menos validez y **dicho como tal** en el reporte.

## Sesgo, dicho con el número en la mano

Doce personas. El modelo aprende de doce fisiologías, una geografía, una dieta y un esquema de
tratamiento; no hay razón para suponer que generaliza a una persona con diabetes tipo 2
insulinodependiente en México. **El reporte dice el número en lugar de esconderlo** detrás de
«un conjunto público reconocido» (riesgo R-2). Lo que lo reduce en parte: el error se desglosa
**por origen del dato**, lo cual permite ver si el sistema funciona peor precisamente para
quien no tiene sensor, que es el usuario con menos recursos.

## Explicabilidad

SHAP por predicción. Las **cinco** contribuciones más grandes, con nombre de variable, valor en
mg/dL y signo, son exactamente lo que consume el servicio de explicación del backend y lo que
se dibuja en la pantalla de factores. **El modelo de lenguaje no inventa la explicación**:
recibe números ya calculados y solo los redacta.

## Lo que el modelo no hace, y hay que dejarlo escrito

No calcula ni sugiere dosis. No diagnostica. No sustituye a la medición: una alerta siempre
pide confirmar antes de actuar. Y **en modo sin sensor no anticipa hipoglucemias**, porque no
tiene con qué.

## Cómo llega el modelo al teléfono

Entrenamiento aquí → artefacto **convertido a Core ML** → se distribuye con la app o se
descarga como recurso versionado → **la inferencia corre en el dispositivo**, sin red, en
milisegundos, y ningún dato de glucosa sale del teléfono para predecir. El backend reentrena
por lotes, publica una versión con sus métricas y la app la adopta en la siguiente
sincronización (RF-29: mensualmente o al detectar deriva del error).

## Estructura

```
glucy-ml/
  data/            vacío en git — el conjunto NO se sube
  src/
    ingest/        lectura de los XML de OhioT1DM
    features/      features.py — los dos juegos, denso y disperso
    train/         train_m1.py, train_m2.py, train_m3.py
    eval/          clarke.py, metrics.py, baselines.py
    export/        to_coreml.py
  notebooks/       cuaderno de evaluación, con la rejilla de Clarke dibujada
  artifacts/       modelos versionados + su JSON de métricas
  tests/
```

## Órdenes de uso diario

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.train.train_m1          # entrena el modelo de 30 minutos
python -m src.eval.metrics --modelo artifacts/m1-v0.1.0.json
pytest -q
```

Toda corrida de entrenamiento **deja su semilla, sus hiperparámetros y sus métricas junto al
artefacto**. Un resultado que no se puede repetir no sirve para un trabajo que se va a
evaluar.

## Reglas clínicas: hay una referencia ejecutable

`referencia/reglas_clinicas.py` del repositorio **glucy-backend** implementa COB, IOB,
indicadores, zonas de Clarke, sensibilidad a hipoglucemia, tasa de falsas alarmas y la línea
base de persistencia, con **30 pruebas pasando**. El código de este repositorio tiene que dar
los mismos números. Nota honesta que ya está escrita ahí: la implementación de la rejilla de
Clarke sigue la formulación habitual, pero conviene cotejarla contra el artículo original de
Clarke et al. (1987) antes de reportar métricas en el documento final.

## Git

Ramas `main` (protegida) y `develop`, más `feature/` y `fix/`. **Conventional Commits en
español.** Un pull request por funcionalidad con qué se veía antes y qué se ve después. Un
issue por requisito, etiquetado con su fase. GitHub Actions corre las pruebas en cada pull
request.

## Dónde está el resto de la documentación

Carpeta `glucy-base` en los archivos del proyecto de Claude: `00-CONTEXTO-GLUCY.md`, `03-glucy-ml.md`
(la tabla de correspondencia con OhioT1DM y el detalle de la validación) y
`05-PENDIENTES-Y-CONTRADICCIONES.md`. La fuente es `Glucy - Reporte del Avance 1.pdf`.

## Aviso que va en el README

Glucy **no es un dispositivo médico certificado**. Los resultados se obtienen sobre doce
sujetos y **no son generalizables a población abierta** (riesgo R-2).
