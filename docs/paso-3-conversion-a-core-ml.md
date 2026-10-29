# Fase 3, paso 3 — la conversión a Core ML, y por qué hubo que comprobarla

El último tramo del camino del modelo: entrenamiento aquí → artefacto convertido a Core ML →
se distribuye con la app → **la inferencia corre dentro del teléfono**, sin red, en
milisegundos, y ningún dato de glucosa sale de ahí para predecir.

```bash
python -m src.export.to_coreml --artefacto artifacts/m1-v0.1.0.json --datos sintetico
```

Este paso **no necesita los datos de OhioT1DM**, igual que los dos anteriores: se escribe
ahora para que el día que lleguen el trabajo sea entrenar y medir.

## Qué cambia

- `src/export/arbol.py` — el bosque que lleva dentro un `.mlmodel`, leído y evaluado en Python
  puro.
- `src/export/to_coreml.py` — la conversión, la comprobación y la orden.
- `tests/test_arbol_de_coreml.py` — 7 pruebas del recorrido del árbol, biblioteca estándar.
- `tests/test_export_coreml.py` — 15 pruebas de la conversión de verdad.

## La conversión son tres líneas; la comprobación es el paso

`coremltools.converters.xgboost.convert` convierte en una llamada. Lo que ocupa este paso es
lo otro: **que el modelo convertido dé los mismos números que el que se midió.** Sin eso, el
reporte presenta las métricas de un modelo y el teléfono ejecuta otro, y no habría forma de
enterarse: un árbol traducido mal no falla, predice mal, y lo que predice es una glucosa.

Y hay un motivo concreto para desconfiar: al convertir, la propia herramienta avisa de que
**la última versión de XGBoost con la que se probó es la 1.4.2**. Aquí se usa la 3.4.1. Dar por
bueno lo que salga sería confiar en una compatibilidad que la herramienta no afirma.

### El problema: `coremltools` convierte en Linux pero no puede ejecutar el modelo

`libcoremlpython` es solo de macOS. Así que la comprobación obvia —«predice lo mismo»— no se
puede hacer donde se convierte, y dejarla «para cuando esté en la Mac» significa no hacerla.

La salida es no ejecutar el modelo sino **leer su especificación**, que es un protobuf con los
nodos del bosque dentro, y recorrerla aquí. `src/export/arbol.py` hace eso en Python puro: si
el recorrido da lo mismo que XGBoost, la traducción es fiel. Lo único que queda fuera es si el
sistema del teléfono recorre el árbol igual, y eso se comprueba una vez en el dispositivo.

## Lo que encontró la comprobación, que es el punto del paso

### 1. El error grave: el modelo del teléfono predecía 143 mg/dL por debajo

Desde XGBoost 2.0, si no se fija `base_score` el modelo **lo aprende de los datos**. Con
glucosa eso es la media de la serie: en la corrida de prueba, **143.72 mg/dL**.
`coremltools` escribe en la especificación un `basePredictionValue` de **0.5**, que era el
valor por omisión de XGBoost 1.x, en lugar de leer el que trae el modelo.

El resultado: **toda predicción del teléfono, 143 mg/dL por debajo de la que se midió.** Una
glucosa real de 140 saldría como −3. Y no falla nada: el archivo se convierte, se guarda, se
carga en el teléfono y predice.

`_con_el_valor_de_base_corregido` lo arregla leyendo el `base_score` de la configuración del
booster y escribiéndolo en la especificación. Hay una prueba que lo fija en las dos
direcciones: que la conversión cruda escribe 0.5, y que la corregida escribe el valor
aprendido.

Un detalle que costó una vuelta: **`MLModel.get_spec()` devuelve una copia.** Modificarla no
modifica el modelo, así que la corrección se perdía sin que nada avisara —y la comprobación
seguía dando los mismos 143 mg/dL de diferencia—. Se arma un `MLModel` nuevo con la
especificación corregida.

### 2. La app tiene que mandar `Float` de 32 bits, no `Double`

Los umbrales del árbol son de 32 bits, y **XGBoost cuantiza cada entrada a 32 bits antes de
comparar**. Quien compare en doble precisión contra esos umbrales voltea algún corte: medido,
hasta **5.03 mg/dL de diferencia en 749 de 1500 filas**.

No es una tolerancia que se pueda ignorar, es una instrucción para el código de Swift: cada
característica se redondea a `Float` antes de llamar al modelo. La comprobación lo mide y lo
escribe en el informe (`la_app_tiene_que_mandar_float_de_32_bits`).

La prueba que lo fija no busca el caso al azar —al azar casi nunca pasa, y «casi nunca» en una
prueba significa que un día falla sola y nadie sabe por qué—: **construye** una fila cuyo valor
en doble precisión está un paso por debajo del umbral y al redondearlo a 32 bits da el umbral
exacto.

### 3. Los valores que faltan: hay algo que comprobar en el iPhone

La regla del proyecto es que **lo que falta es `NaN` y nunca cero**, porque XGBoost aprende para
qué lado mandar un `NaN` y el cero le arrastra el promedio. En la corrida de prueba, **709 de
4124 cortes mandan el faltante al hijo verdadero**.

La buena noticia: la conversión **no pierde esa dirección**. La especificación marca
`missingValueTracksTrueChild` en exactamente los mismos cortes que el volcado de XGBoost, y hay
una prueba que compara las dos cuentas.

La que queda abierta: **un sistema que no entienda esa marca y compare el `NaN` como cualquier
número se iría siempre por el hijo falso**, porque toda comparación IEEE con `NaN` da falso.
Medido: hasta **23.3 mg/dL de diferencia**, con 86 de 1500 filas moviéndose más de 5 mg/dL. Eso
no se puede comprobar desde Linux. El informe lo marca
(`hay_que_comprobar_el_faltante_en_el_iphone`) y **se comprueba una vez en el dispositivo**:
una fila con `NaN`, el número de Core ML contra el número de este repositorio.

## Lo que hace el exportador, además de convertir

- **Si la traducción no es fiel, no escribe nada.** Un `.mlmodel` que no predice lo que se
  midió es peor que no tenerlo: con el archivo en la carpeta, alguien lo mete a la app.
- **No sobrescribe**, por lo mismo que los artefactos (RF-25): cada predicción guarda con qué
  versión se calculó, y pisar un `.mlmodel` rompe ese rastro sin avisar.
- **El informe se arma antes de escribir**, para que un fallo al serializarlo no deje un
  `.mlmodel` a medias que además bloquearía el reintento.
- **Las características en su orden viajan dentro del archivo**, en los metadatos, junto con la
  unidad y el aviso de que Glucy no es un dispositivo médico. El orden es el contrato con
  Swift: una columna corrida da números plausibles y equivocados sin que nada falle.
- **El archivo dice si sus métricas se pueden reportar** (`glucy.se_puede_reportar`), con el
  mismo criterio del manifiesto. Es la protección contra un artefacto de corrida sintética
  llegando al teléfono o al documento sin que nadie lo note.
- **Las filas de comprobación se arman con el mismo código que entrena.** Comprobar con
  números al azar sería más fácil y mucho peor: no tendrían `NaN` donde la serie real los
  tiene, y el único caso donde esta conversión puede desviarse no se tocaría nunca.

## Un error del comprobador que vale contar

La primera versión decía que la conversión estaba mal por **90 mg/dL de media**. No lo estaba:
el comprobador tenía el orden del enumerado al revés. En el protobuf de Core ML los
comportamientos de un corte son `<=`, `<`, `>=`, `>` en ese orden —los «menor que» **antes** que
los «mayor que»—, que no es el orden que uno supondría, y con la tabla invertida cada corte del
árbol se recorre al revés.

Está escrito como prueba (`test_los_numeros_del_comportamiento_no_son_los_intuitivos`) y los
números se leyeron del protobuf, no de la memoria. La lección es la del paso: **la primera
diferencia grande puede estar en quien mide**, y hay que descartarlo antes de tocar lo medido.

## Lo que este paso no hace

- **SHAP no se toca aquí, y no hace falta**: el backend calcula las contribuciones con
  `pred_contribs` del propio XGBoost, sobre el artefacto `.ubj`, y el servicio de explicación
  ya las consume. Lo que falta de explicabilidad es el cuaderno de evaluación.
- **Core ML no da contribuciones.** Un bosque convertido predice un número y nada más, así que
  **la predicción que corre en el teléfono no trae SHAP consigo**. La pantalla de factores
  necesita el `/explain` del backend, o calcular TreeSHAP en el dispositivo sobre la misma
  estructura de árboles. Está dicho aquí porque es una decisión de arquitectura que todavía no
  se tomó, no un pendiente de código.
- **No se comprueba en el dispositivo.** Las dos cosas que faltan —si Core ML respeta la marca
  del faltante y si los números coinciden en el iPhone— son de la Mac y del teléfono.
- Sigue faltando la **línea base LSTM**, y el reporte tiene que decirlo.

## Órdenes para comprobar este paso

```bash
pytest -q                                            # 227 pruebas
ruff check .
python -m src.train.train_m1 --datos sintetico --sujetos 4 --dias 6 --version v0.0.1
python -m src.export.to_coreml --artefacto artifacts/m1-v0.0.1.json --datos sintetico
```

La segunda orden deja el `.mlmodel` y su informe junto al artefacto. Con las series de juguete
el informe sale con `origen_de_las_filas_de_comprobacion: "sintetico"`, que es lo correcto:
dice que **la traducción** es fiel, no que el modelo sirva.
