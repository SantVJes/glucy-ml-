# Paso 2 de la fase 3 — el esqueleto del entrenamiento

Fecha: 29 de octubre de 2026 · Rama: `feature/esqueleto-de-entrenamiento`

---

## Qué cambia

**Antes:** el repositorio tenía con qué medir —rejilla de Clarke, métricas, particiones,
características— y nada con qué entrenar. Los tres modelos del plan, M-1, M-2 y M-3, no
existían ni como archivo.

**Después:** los tres guiones corren de punta a punta, con validación cruzada dejando un sujeto
fuera, cuatro líneas base medidas en los mismos pliegues y un artefacto versionado que guarda
semilla, hiperparámetros, métricas y de dónde salieron los datos. 205 pruebas pasando.

**El día que llegue OhioT1DM, entrenar es un comando:**

```bash
python -m src.train.train_m1    # glucosa a 30 min
python -m src.train.train_m2    # glucosa a 60 min
python -m src.train.train_m3    # pico posprandial y cuándo ocurre
```

Y hoy, sin el conjunto, el mismo comando corre con `--datos sintetico`.

## Lo que se agregó

```
src/features/ventanas.py    de una serie a las filas que entrenan, con su objetivo
src/datos/sintetico.py      series de juguete, para poder correr todo sin el conjunto
src/train/modelos.py        XGBoost, Ridge, Random Forest y las dos líneas base triviales
src/train/entrenar.py       el bucle LOSO: modelo y líneas base en los mismos pliegues
src/train/artefacto.py      el artefacto versionado y su manifiesto
src/train/comun.py          lo que comparten los tres guiones
src/train/train_m1.py       M-1, M-2 y M-3
src/train/train_m2.py
src/train/train_m3.py
```

## Las decisiones que el diff no explica solo

### 1. Los rezagos se buscan por tiempo, nunca por posición

`rezago_30` es «la lectura más cercana a hace 30 minutos, si existe dentro de dos minutos y
medio», no «la fila de seis atrás». HealthKit entrega lo que la app del fabricante escribió,
cuando lo escribió (RF-06b), y OhioT1DM trae huecos. Buscar por posición devuelve un valor de
otro momento con el nombre de este, y nada falla.

Lo mismo con el objetivo: **si no hay lectura cerca de los 30 minutos, esa fila se descarta**
en lugar de usar la más próxima que haya. Un objetivo a 52 minutos etiquetado como «30
minutos» es ruido con apariencia de dato. Es la decisión que más filas tira en un conjunto con
huecos, y es la correcta.

### 2. Una fila no puede mirar hacia adelante, y hay una prueba que lo demuestra

Cada fila se arma con `serie[:i+1]` y nada más. La prueba que lo comprueba recorta la serie
después del punto que se predice y exige **exactamente el mismo vector**: si alguna
característica se colara desde el futuro, cambiaría.

Importa porque mirar adelante es el error que produce las métricas más espectaculares y más
falsas, y no se ve leyendo el código que lo causa. En producción no existe: a las 14:00 el
teléfono no tiene las lecturas de las 14:05.

### 3. La ventana de la pendiente cambia la extrapolación por un factor de tres

Al medirlo salió esto, extrapolando a 30 minutos:

| Ventana de la pendiente | RMSE de la línea base |
| --- | --- |
| 5 min (`derivada_1`) | 53.1 |
| 15 min | 26.5 |
| **30 min (la que se usa)** | **20.0** |
| 60 min | 17.9 |

La pendiente de cinco minutos es casi toda ruido del sensor, y multiplicarla por treinta lo
amplifica por seis. **Y una línea base pésima adula al modelo**, que es justo la dirección en
la que no conviene equivocarse: el proyecto se sostiene en «le gana a la persistencia y a la
extrapolación», y ganarle a una versión mal hecha de la segunda no demuestra nada.

Por omisión la ventana es igual al horizonte, que es lo habitual en la literatura y lo
defendible: «el ritmo que lleva» medido en el mismo plazo que se proyecta. El valor queda en
los hiperparámetros del manifiesto para que el reporte pueda decir cuál se usó.

### 4. El modelo y las líneas base se miden en los mismos pliegues, en el mismo bucle

«RMSE de 18 mg/dL» no significa nada suelto. La forma de acabar comparando contra un número de
otra corrida es tener el modelo en un archivo y las líneas base en otro, así que aquí entran al
mismo bucle y el manifiesto guarda las métricas de las líneas base **al lado** de las del
modelo.

Tres detalles del bucle, cada uno con su prueba:

- **Un modelo nuevo por pliegue.** Reutilizar el objeto lo seguiría entrenando con los
  pliegues anteriores: el del último pliegue habría visto a los doce sujetos y el resultado
  saldría inflado sin que nada falle.
- **Las métricas se agregan sobre todas las predicciones juntas**, no promediando los doce
  números por pliegue. Promediar porcentajes de muestras de distinto tamaño los pondera mal: un
  sujeto con 400 filas acabaría pesando lo mismo que uno con 4 000.
- **Y además por sujeto**, siempre. Con doce personas el promedio esconde a aquella para la que
  el sistema funciona peor, y eso es exactamente lo que el riesgo R-2 pide poder mirar.

### 5. «Le gana a la persistencia» exige dos cosas, no una

```python
self.resultado.rmse < persistencia.rmse and self.resultado.clarke_ab >= persistencia.clarke_ab
```

Un RMSE mejor con más predicciones en zonas peligrosas no es mejor para una persona. La
comprobación ya hizo su trabajo: en la corrida de prueba, XGBoost sacó mejor RMSE que la
persistencia (13.75 contra 15.43) y **peor** A+B (97.83 % contra 99.23 %), así que el guion
imprimió el aviso de que el modelo no justifica su existencia. Con un solo criterio habría
dicho que iba bien.

### 6. El artefacto se entrena con todos los sujetos; las métricas salen de los pliegues

Los pliegues sirven para **medir**, no para elegir el modelo que se publica. Publicar el de un
pliegue sería publicar un modelo que nunca vio a una de las doce personas, por nada.

Eso deja el manifiesto con `sujetos_prueba` vacío, que se leería como «no se evaluó», así que
el manifiesto lleva un campo `protocolo` que lo explica en una línea. Un número sin su
protocolo al lado no se puede leer.

### 7. Los artefactos no se sobrescriben

Cada predicción guarda con qué versión del modelo se calculó (RF-25), y la tabla `modelos` del
backend también. Pisar `m1-v0.1.0.json` rompe ese rastro sin avisar, así que guardar sobre uno
que ya existe revienta y pide subir la versión.

### 8. El origen de los datos va en el manifiesto, y decide si las métricas se pueden reportar

`ohiot1dm`, `simulador` o `sintetico`. Con las series de juguete, `se_puede_reportar` es falso
y el guion lo imprime en pantalla. Es la protección contra el peor error posible de este
repositorio: un número de una corrida sintética llegando al documento final sin que nadie
mintiera a propósito. Por lo mismo, `--datos ohiot1dm` **revienta si la carpeta está vacía** en
lugar de caer al generador.

## Los tres errores que encontraron las pruebas

Vale la pena anotarlos porque los tres estaban en mi propio código y los tres son del tipo que
no se ve leyendo:

1. **El nombre del artefacto perdía la versión de parche.** `Path.with_suffix(".json")` trata
   el `.0` de `m1-v0.1.0` como extensión y lo reemplaza, así que el manifiesto salía como
   `m1-v0.1.json`: con un nombre distinto del `.ubj` del modelo y pisando el de cualquier otra
   versión de parche de la misma menor. Hay una prueba que lo fija.
2. **Mis filas de prueba eran irreales, no los modelos.** Tenían columnas enteras en `NaN`, y
   con eso `HistGradientBoostingRegressor` revienta y el imputador de Ridge avisa de que no
   tiene con qué imputar. Las filas de verdad traen `NaN` en algunas filas, no en columnas
   completas. Se corrigieron las pruebas.
3. **Una prueba comparaba dos personas sintéticas por su promedio** y fallaba por casualidad:
   un nivel de base más alto se cancela con una sensibilidad a la insulina más alta. Ahora mira
   la dispersión de las doce, que es lo que de verdad hace que repartir por persona signifique
   algo.

## Lo que falta, dicho con nombre

- **La línea base LSTM.** El protocolo pide cinco y hay cuatro. Necesita un marco de redes
  —PyTorch o Keras—, que son cientos de megabytes y una decisión que conviene tomar con los
  datos delante: con doce sujetos y ocho semanas cada uno puede que una red recurrente no se
  sostenga, y eso se sabe midiendo. **Mientras no esté, el reporte tiene que decir que falta**,
  no omitirla de la tabla.
- **SHAP.** Las cinco contribuciones por predicción, que son lo que consume el servicio de
  explicación del backend. Van después del primer modelo entrenado con datos de verdad, porque
  explicar un modelo entrenado con senos no sirve para nada.
- **La conversión a Core ML.** Igual: necesita un artefacto que valga la pena convertir.
- **La búsqueda de hiperparámetros.** Los de arranque son conservadores a propósito. La
  búsqueda va **solo contra validación**, nunca mirando la prueba, y por eso no está en este
  bucle: ajustarlos mirando la prueba equivale a entrenar con la prueba, y el número que sale
  de ahí no se puede reportar.
- **El cuaderno de evaluación** con la rejilla de Clarke dibujada.

## Sobre las series de juguete, y lo que no son

`src/datos/sintetico.py` **no es un simulador fisiológico**. Es una suma de senos con el efecto
de las comidas y de la insulina encima, y existe para una sola cosa: que un guion falle hoy por
un error de programación y no el día que lleguen los datos.

Si el acceso a OhioT1DM no llega, el respaldo del proyecto es un **simulador fisiológico de
código abierto**, con menos validez y dicho como tal en el reporte. No es este archivo. Una
métrica calculada sobre estas series no significa nada, y el manifiesto lo marca.

Lo que sí hacen bien, porque de eso dependen las pruebas: son reproducibles con su semilla, las
doce personas son distintas entre sí, y la mezcla tiene hipoglucemias, hiperglucemias y huecos
—alrededor de 74 % del tiempo en rango, 1.7 % por debajo de 70 y picos de 278 mg/dL—, que son
los tres casos que el resto del repositorio existe para manejar.

## La integración continua sigue instalando solo pytest y ruff

A propósito, y conviene no cambiarlo. Todo lo que sostiene las reglas clínicas —la rejilla, las
métricas, las particiones, las características, las ventanas, el manifiesto y las dos líneas
base triviales— está escrito con biblioteca estándar. Las pruebas que necesitan xgboost o
scikit-learn se saltan solas cuando no están.

Comprobado en las dos direcciones: **205 pruebas con todo instalado, 199 pasando y 6 saltadas
ocultando las bibliotecas**, y la suite tarda medio segundo. Si dependiera de una rueda de 200
megabytes, dejaría de correr en cada pull request, que es justo cuando vale.

## Órdenes para comprobar este paso

```bash
ruff check . && ruff format --check .
pytest -q                                           # 205 pruebas
python -m src.train.train_m1 --datos sintetico --sujetos 6 --dias 6 --sin-guardar
python -m src.train.train_m3 --datos sintetico --sujetos 6 --dias 10
```
