# Traspaso: cómo seguir con glucy-ml sin el contexto de quien lo escribió

Este documento es para quien llegue al proyecto sin haber estado en él, sea una persona u otro
asistente de programación. Junto con `CLAUDE.md` debería bastar para continuar. Hay uno igual
en `glucy-backend` y otro en `glucy-ios`, cada uno con lo de su repositorio.

Escrito el 20 de noviembre de 2026. Si lees esto mucho después, compara contra el historial de
`develop` antes de dar algo por cierto.

## 1. Antes de tocar nada

- **El trabajo vive en `develop`, no en `main`.** `main` se quedó en la fase 0 y nunca se le
  fusionó `develop`. Clona y cambia de rama: `git checkout develop`.
- **No hay etiquetas ni releases** en este repositorio.
- **`main` no está protegida**, aunque `CLAUDE.md` diga que sí. Es un pendiente (issue abierto).
- La carpeta `glucy-base` que cita `CLAUDE.md` está en los archivos privados del proyecto, no
  en GitHub. No hace falta: lo que importa para el código ya está en `CLAUDE.md`, en `docs/` y
  en `glucy-backend/referencia/reglas_clinicas.py`.
- **Los datos de OhioT1DM no están ni deben estar en el repositorio** (acuerdo de uso). Van en
  `data/ohiot1dm/{2018,2020}/{train,test}/*.xml`, ignorado por git. Hay que pedírselos al
  dueño del proyecto. Sin ellos se puede trabajar con el simulador (`docs/paso-7-*.md`) o con
  la serie sintética de las pruebas.

## 2. Dónde está cada cosa

| Qué | Dónde |
| --- | --- |
| Reglas del proyecto, modelos, protocolo de validación, metas | `CLAUDE.md` |
| Cómo se llegó a cada parte, con sus trampas | `docs/paso-1-*.md` a `docs/paso-8-*.md`, en orden |
| Resultados con OhioT1DM | `docs/paso-8-entrenar-con-ohiot1dm.md` |
| Artefactos entrenados con su manifiesto | `artifacts/` (`.json` = manifiesto, `.ubj` = XGBoost, `.mlmodel` = Core ML) |
| Conversión a Core ML y su comprobación | `src/export/to_coreml.py`, `docs/paso-3-conversion-a-core-ml.md` |
| Cuaderno de evaluación con la rejilla de Clarke | `notebooks/`, `docs/paso-6-cuaderno-de-evaluacion.md` |

## 3. Estado al 20 de noviembre de 2026

- M-1, M-2 y M-3 entrenados con OhioT1DM (`v0.3.0-ohiot1dm`), validación dejando un sujeto
  fuera, con cuatro de las cinco líneas base.
- **Ninguno llega a todas sus metas, y eso se reporta tal cual.** M-1: RMSE 20.48 mg/dL (meta
  20), A+B de Clarke 98.09 %, sensibilidad a hipoglucemia 27 % (meta 85), falsas alarmas
  26.6 % (meta 15). M-3: error del pico 40.45 mg/dL (meta 25). El detalle está en `paso-8`.
- M-1 pesa casi todo en «dónde estás ahora» y «dónde estabas hace 5 min»; COB e IOB pesan
  menos del 1 %. Lo que le falta son características de comida e insulina, no hiperparámetros.
- 313 pruebas. La integración continua se salta las que necesitan las bibliotecas pesadas
  (XGBoost, SHAP, coremltools, PyTorch); córrelas en local con `pytest -q`.

## 4. Lo que falta, en el orden en que conviene hacerlo

Cada punto tiene su issue en este repositorio.

1. **Hecho el 24 de noviembre de 2026: `v0.3.0-ohiot1dm` convertido a Core ML** y llevado a la
   app (glucy-ios, `Glucy/Resources/Modelos/`). Fiel a XGBoost: 0.000305 mg/dL de diferencia
   máxima en M-1 y 0.000275 en M-2, sobre 2 000 filas de OhioT1DM. Para la próxima versión:
   ```bash
   python -m src.export.to_coreml --artefacto artifacts/m1-vX.json --datos ohiot1dm \
       --carpeta data/ohiot1dm/OhioT1DM
   ```
   (la carpeta es la que deja el zip al descomprimirse), se copian los `.mlmodel` como
   `GlucyM1.mlmodel` y `GlucyM2.mlmodel`, y se actualizan los números fijos de
   `glucy-ios/GlucyTests/ModeloCoreMLTests.swift` corriendo el `.ubj` con la misma fila en
   `float32`.
2. **Correr la línea base LSTM con OhioT1DM.** El protocolo exige las cinco líneas base y el
   reporte dice que falta esta. Con las otras cuatro la corrida tardó 1 h 50 min; quita
   `--sin-lstm` y córrela donde haya tiempo (mejor con GPU). Agrega la fila a la tabla de
   `paso-8` y al manifiesto.
3. **Mejorar M-1 si da tiempo**: características de comida e insulina (por ejemplo COB e IOB
   con más historia o su tasa de cambio). Cualquier característica nueva **se agrega a la vez
   en el pipeline del backend y en `VectorDeCaracteristicas.swift` de la app**, con el mismo
   nombre. Si no, la predicción se alimenta con columnas cambiadas de lugar y nada falla.
4. Llenar `docs/architecture.md` y `docs/ai.md`, que siguen diciendo «Pendiente de llenar».

## 5. Trampas que ya costaron horas

- **Cada persona de OhioT1DM viene en dos archivos** (train y test). Leídos por separado, la
  misma persona queda de los dos lados de la validación. `leer_conjunto` + `unir` en
  `src/ingest/ohiot1dm.py` lo resuelve; no lo cambies.
- **`coremltools` pierde el `base_score`** aprendido de XGBoost y toda predicción sale unos
  140 mg/dL abajo. `to_coreml.py` lo corrige. `MLModel.get_spec()` devuelve una copia:
  modificarla no modifica el modelo.
- **La app tiene que mandar `Float` de 32 bits**, no `Double`, o la predicción se desvía
  varios mg/dL.
- **La fila se arma en el instante de la última lectura, no con el reloj.** Si se arma con el
  reloj, casi la mitad de las veces los rezagos caen en la lectura equivocada y el error se
  multiplica por diez sin que falle nada.
- **Lo que falta es `NaN`, nunca cero.** XGBoost aprendió hacia dónde mandar el faltante.
- **`sensibilidad_hipo` puede ser `null`** en el manifiesto cuando el conjunto no traía
  hipoglucemias. Así tiene que quedar; un 100 % ahí sería mentira.
- `coremltools` convierte en Linux pero no puede ejecutar el modelo. `src/export/arbol.py`
  recorre el bosque en Python para comprobar la conversión.
- Un contrato entre dos repositorios solo se da por bueno cuando se corrió con un archivo
  producido por el otro lado, no con uno escrito a mano.

## 6. Cómo se trabaja

- Todo en español: código, comentarios, commits (Conventional Commits), ramas y PR.
- Un PR por cosa, **siempre contra `develop`, nunca apilado sobre otro PR**. Una vez se
  fusionaron dos PR apilados casi a la vez y uno acabó en una rama que ya nadie usaba.
- Antes de fusionar, la integración continua tiene que terminar en verde. «Cancelado» y
  «saltado» no son verde.
- Toda corrida de entrenamiento deja semilla, hiperparámetros y métricas en su manifiesto.

## 7. Aviso

Glucy no es un dispositivo médico certificado. Los resultados salen de doce personas con
diabetes tipo 1 y no son generalizables a población abierta (riesgo R-2).
