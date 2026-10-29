# Paso 7 — entrenar con el simulador fisiológico, y la comparación que estaba mal hecha

Este es el primer paso de la fase 3 que **entrena de verdad**. Los seis anteriores escribieron
la evaluación, las características, el esqueleto del entrenamiento, la conversión a Core ML, la
quinta línea base y el cuaderno, todos sin datos y a propósito: para que el día que hubiera
datos el trabajo fuera entrenar y medir, y no decidir cómo se mide.

El conjunto real, OhioT1DM, no ha llegado. El correo se envió el **29 de octubre de 2026** y
el proceso tarda hasta una semana, con corte el **8 de noviembre**. Así que se activó el plan B
que el reporte ya prometía por escrito: el **simulador fisiológico**.

## Por qué no servía lo que ya había

`src/datos/sintetico.py` es un generador de series de juguete, y existe para probar que los
guiones corren. Entrenado con él, M-1 dio esto:

| | RMSE | A+B | Sens. hipo |
| --- | --- | --- | --- |
| M-1 | 11.89 | 99.44 % | 48.1 % |
| persistencia | 14.93 | **99.49 %** | 55.8 % |
| extrapolación | 19.76 | 99.66 % | 78.3 % |

Un 99.4 % de zonas A+B parece un resultado excelente hasta que se mira la fila de abajo: **la
línea base más tonta saca más**. Las seis filas pasaban del 98 %, incluida la recurrente. Eso
no mide un modelo bueno, mide una señal que es predecible por construcción. Los dos avisos del
guion saltaron solos: «no le gana a la persistencia» y «estas métricas NO se pueden reportar».

## El simulador: simglucose, modelo UVA/Padova 2008

```bash
pip install "setuptools<81" simglucose
python -m src.datos.simulador --dias 28 --salida data/simulador
python -m src.train.train_m1 --datos simulador
```

**Trampa al instalar, porque el mensaje de error no apunta a la causa:**
`simglucose/__init__.py` importa gym 0.9.4, que hace `import pkg_resources`, y **setuptools 81
lo quitó**. Sin el tope, `import simglucose` falla con `ModuleNotFoundError: No module named
'pkg_resources'` y nada en el mensaje menciona setuptools. De ahí el `setuptools<81` en
`requirements.txt`.

### 18 de los 30 sujetos, y la lista no está escrita a mano

simglucose trae 30 pacientes virtuales: diez niños, diez adolescentes y diez adultos. Se usan
**18**. Los diez niños tienen entre 7 y 12 años, y `adolescent#006` y `adolescent#008` tienen
14: por debajo de la **edad mínima de 15** del perfil de Glucy. Entrenar con fisiologías de
niños de siete años para una app que no admite menores de quince es desajuste entre el
entrenamiento y el uso.

Lo que importa aquí no es el número, es que la lista **se comprueba sola**.
`tests/test_simulador.py` la coteja contra las edades del propio `params/Quest.csv` de
simglucose **en los dos sentidos**: que no sobre nadie, y que no falte nadie. Una lista escrita
a mano se desincroniza en silencio, y con doce personas en el conjunto real cada sujeto perdido
sin querer es un ocho por ciento del conjunto.

### Sensor GuardianRT y no Dexcom

| Sensor de simglucose | Muestreo |
| --- | --- |
| Navigator | 1 min |
| Dexcom | 3 min |
| **GuardianRT** | **5 min** |

Toda la tubería de Glucy supone cinco minutos: los rezagos a 5/10/15/30/60 min, la
interpolación de huecos de hasta 15 min y el umbral de frescura del sensor. Con el Dexcom
habría que remuestrear, y remuestrear introduce un error que después no se puede separar del
error del modelo.

### Bucle propio en vez de `show_history()`

`T1DSimEnv.show_history()` devuelve una columna `insulin` que es **la suma de basal y bolo**.
Glucy los distingue —RF-14, y `HKMetadataKeyInsulinDeliveryReason` en HealthKit— y el IOB solo
cuenta bolo, así que sumarlos falsearía el IOB de todas las filas. De ahí el bucle propio en
`src/datos/simulador.py`, que lee `accion.basal` y `accion.bolus` por separado en cada paso.

### La semilla depende del sujeto, no del orden

`uno(sujeto)` calcula la semilla desde la **posición del sujeto en `SUJETOS`**, y `conjunto()`
delega en ella. Así `--solo adult#005` produce exactamente la misma serie que la tanda
completa, y la tanda se puede repartir en varios procesos sin que el conjunto cambie. Si la
semilla dependiera del orden de la llamada, el conjunto en disco dependería de con cuántos
procesos se generó, que es la clase de detalle que nadie vuelve a mirar.

Simular los 18 sujetos lleva unos 22 minutos repartido en cuatro procesos.

### Se simula una vez; los tres entrenamientos leen de disco

`guardar` deja un JSON por sujeto en `data/simulador/`, que no se versiona porque se regenera
de la semilla. Los tres guiones **leen**, no simulan. Si cada uno simulara por su cuenta, M-1,
M-2 y M-3 podrían acabar medidos sobre tres conjuntos distintos sin que nada fallara, y sus
métricas dejarían de ser comparables entre sí.

`cargar` rechaza un JSON cuyo campo `origen` no sea `simulador`. El manifiesto del modelo
existe para que ningún número llegue al documento final sin su etiqueta; un archivo ajeno leído
aquí saldría etiquetado como del simulador y la etiqueta dejaría de valer.

## La limitación, dicha aquí para que esté en el reporte

El controlador basal-bolo del simulador **calcula el bolo con el conteo exacto de carbohidratos
de la comida**. Una persona real estima a ojo, y ese error de conteo es una de las fuentes
principales de variabilidad en la glucosa posprandial. La variabilidad del simulador es por
tanto **menor que la real**, y un modelo entrenado aquí ve un problema más fácil que el de
verdad.

**No se le añadió un modelo de error de conteo.** Sería un número inventado que el simulador no
sostiene, y el proyecto preferiría reportar una limitación a reportar una corrección de
procedencia desconocida.

Lo que sí hay, medido sobre 28 días por sujeto y 145 152 lecturas en total, es hipoglucemia de
sobra y **muy distinta entre personas**:

| | % del tiempo < 70 mg/dL |
| --- | --- |
| `adult#003` | 0.0 % (ninguna) |
| mediana de los 18 | 2.8 % |
| `adult#009` | 21.1 % |

Esa dispersión es lo que hace que repartir por sujeto signifique algo: con dieciocho personas
idénticas, dejar un sujeto fuera y repartir al azar darían lo mismo.

## El hallazgo del paso: la sensibilidad estaba mal comparada

M-1 sobre el simulador, 145 062 filas, 18 pliegues y **las cinco líneas base** del protocolo:

| | RMSE | MAE | MARD | A+B | Sens. hipo | Falsas |
| --- | --- | --- | --- | --- | --- | --- |
| M-1 | **14.06** | **10.52** | **8.75 %** | 98.96 % | 59.2 % | 20.2 % |
| persistencia | 19.64 | 14.08 | 11.15 % | 99.08 % | **67.8 %** | 32.1 % |
| extrapolación lineal | 24.14 | 17.84 | 14.38 % | 99.16 % | 75.3 % | 58.4 % |
| ridge | 14.88 | 11.13 | 9.36 % | 98.46 % | 45.9 % | 17.4 % |
| bosque | 14.24 | 10.60 | 8.77 % | 98.96 % | 59.5 % | 20.7 % |
| lstm | 15.94 | 11.83 | 9.94 % | 97.58 % | 29.3 % | **8.5 %** |

Cumple cuatro metas de seis: RMSE ≤ 20, MAE ≤ 15, MARD ≤ 12 % y A+B ≥ 95 %. **Falla las dos
clínicas**: sensibilidad 59.2 % contra una meta de 85 %, y falsas alarmas 20.2 % contra 15 %.

M-1 tiene **el mejor RMSE, MAE y MARD de las seis filas**, la recurrente incluida. Lo único que
no gana es el A+B de Clarke, por 0.12 puntos contra la persistencia.

Y la persistencia caza más hipoglucemias que el modelo. El guion lo detectó solo, porque «le
gana a la persistencia» exige mejor RMSE **y** no peor A+B, y el A+B es 0.12 puntos peor.

### Por qué esa comparación no valía

Mirando las dos filas con cuidado: el modelo tiene **menos** sensibilidad **y menos** falsas
alarmas. Dos números que bajan a la vez no dicen cuál es mejor; dicen que están medidos en
**puntos de operación distintos**.

Un modelo de regresión entrenado con error cuadrático **encoge hacia la media**, porque encoger
es lo óptimo para el RMSE. Encogido, cruza el umbral de 70 menos veces: avisa menos, así que se
le escapan más episodios y también se equivoca menos cuando avisa. La persistencia copia el
valor actual, no encoge nada, avisa de más, caza más y falla más. El umbral de aviso estaba
fijo en 70 para los dos, que es el **umbral clínico**, y usar el mismo número para definir la
enfermedad y para disparar el aviso mezcla dos cosas distintas.

`src/eval/umbral_de_aviso.py` las separa. El clínico no se mueve —hipoglucemia son 70 mg/dL y
punto, moverlo sería cambiar la definición de la enfermedad para que el modelo la acierte— y el
de aviso barre de 55 a 110 mg/dL. Con eso se responde la pregunta que sí vale: **cuánto caza
cada uno cuando los dos molestan igual de seguido.**

La comparación se imprime en cada corrida y no detrás de una bandera, porque el cuadro de
arriba por sí solo invita a la conclusión equivocada.

### El resultado, que da la vuelta a la conclusión

```
línea base con aviso < 70 mg/dL:            sensibilidad 67.8 %   falsas 32.1 %
modelo igualando esas falsas (aviso < 75):  sensibilidad 73.5 %   falsas 31.1 %
```

**A igual molestia el modelo caza 5.7 puntos más de hipoglucemias.** El 59.2 % contra 67.8 %
del cuadro de arriba no decía que el modelo fuera peor cazando hipoglucemias; decía que avisaba
menos. Puesto a avisar igual de seguido que la persistencia, caza más.

Y aun así `le_gana_a_la_persistencia` sigue en **falso**, porque exige mejor RMSE y no peor
A+B, y el A+B es 0.12 puntos peor. Con las seis filas por encima del 98 % de A+B, ese criterio
está decidiendo con la diferencia menos informativa del cuadro, y eso se dice en el reporte en
lugar de ajustar el criterio hasta que dé verdadero.

Lo que sigue fallando, y hay que decirlo sin adornos: ni al umbral igualado ni al que más caza
se llega al **85 % de sensibilidad** con **15 % o menos de falsas alarmas**. Ningún umbral
cumple las dos metas a la vez con estas características.

## M-2, a 60 minutos

```
                             n     RMSE      MAE    MARD     A+B  Sens.hipo   Falsas
m2                      144954    22.05    16.05  13.21%  97.35%      34.7     27.7%
persistencia            144954    30.98    21.99  17.21%  97.85%      55.6     44.4%
extrapolación lineal    144738    44.34    31.38  24.28%  96.34%      76.6     72.8%
ridge                   144954    23.56    17.21  14.48%  95.94%       6.1     26.7%
bosque                  144954    22.34    16.17  13.26%  97.39%      35.4     29.5%
lstm                    144954    23.98    17.11  14.06%  95.68%       9.5     15.9%
```

Igualando la molestia: la persistencia caza 55.6 % con 44.4 % de falsas; M-2 con aviso < 79
mg/dL caza **57.8 %** con 43.1 %. Otra vez caza más a igual molestia, por 2.3 puntos, y otra
vez el A+B es el que decide que «no le gana» —0.50 puntos esta vez.

Lo esperable y lo revelador, en ese orden. Lo esperable: a 60 minutos todo empeora, porque hay
el doble de futuro por delante; el RMSE pasa de 14.06 a 22.05 y la persistencia se degrada
mucho más rápido (19.64 → 30.98), que es exactamente lo que justifica tener dos modelos
independientes por horizonte (RF-26) en vez de uno extrapolado.

Lo revelador es la fila de **ridge**: 6.1 % de sensibilidad. Un modelo lineal a 60 minutos
encoge tanto hacia la media que casi nunca cruza el umbral, y se le escapan diecinueve de cada
veinte hipoglucemias **con un RMSE de 23.56 que se ve respetable**. Es el argumento entero de
este paso en una sola fila: el RMSE no mide lo que la app necesita medir.

## La línea base recurrente, y lo que su fila enseña

La LSTM entra en los dos cuadros de arriba y en el manifiesto de los dos artefactos. Las
primeras corridas se habían hecho con `--sin-lstm` porque con las cinco pasaban de dos horas y
media sin terminar; con el arreglo de rendimiento que salió de este mismo paso, **una corrida
con las cinco tarda nueve minutos**, así que ya no hay nada que explicar en el reporte.

Y su fila dice algo que el resto del cuadro no decía tan claro. En M-1 la recurrente tiene
**8.5 % de falsas alarmas: cumple la meta de ≤ 15 % y es la única fila que la cumple.** También
tiene 29.3 % de sensibilidad, o sea que **se le escapan siete de cada diez hipoglucemias**.

Puestas en orden de cuánto encoge cada modelo hacia la media, las dos columnas clínicas se mueven
juntas y en direcciones opuestas:

| | Sens. hipo | Falsas |
| --- | --- | --- |
| extrapolación lineal (no encoge nada) | 75.3 % | 58.4 % |
| persistencia | 67.8 % | 32.1 % |
| bosque | 59.5 % | 20.7 % |
| M-1 | 59.2 % | 20.2 % |
| ridge | 45.9 % | 17.4 % |
| lstm (encoge más) | 29.3 % | **8.5 %** |

**La meta de falsas alarmas se cumple sola con un modelo que casi no avisa**, y la de
sensibilidad se cumple sola con uno que avisa casi siempre. Por separado no significan nada; por
eso `Punto.cumple_las_metas` exige las dos a la vez y por eso la comparación válida es a igual
tasa de falsas alarmas. Una tabla que reportara solo la columna de falsas alarmas premiaría al
peor modelo de los cinco.

A 60 minutos el efecto llega al extremo: ridge caza **6.1 %** y la recurrente **9.5 %**, con
RMSE de 23.56 y 23.98 que se ven respetables al lado del 22.05 de M-2.

### La parte que el código no puede hacer cumplir

Elegir el umbral mirando las métricas del pliegue de prueba y reportar después esas mismas
métricas **es entrenar con la prueba**, y desde el código no hay forma de notarlo: los números
se ven iguales. Por eso `curva` devuelve la curva entera, para dibujarla y reportarla, y
`umbral_que_cumple_las_metas` recibe los datos sobre los que está permitido elegir. Su
docstring dice que son los de entrenamiento. El reporte tiene que decirlo también.

## M-3, el que sí cumple

El modelo del pico posprandial, 1 858 comidas de 18 sujetos:

| | n | RMSE | MAE |
| --- | --- | --- | --- |
| Valor del pico | 1 858 | 29.82 mg/dL | **22.52 mg/dL** |
| Cuándo ocurre | 1 858 | 36.82 min | 27.86 min |

**Cumple su meta**, que es ≤ 25 mg/dL de MAE en el valor del pico. Es el primer modelo de Glucy
que pasa su criterio. El «cuándo ocurre» no tiene meta fijada en el reporte y se reporta tal
cual.

Lo que no puede faltar al lado de ese número: el error **por persona** va de 14.07 mg/dL en
`adolescent#009` a 44.07 en `adolescent#007`, tres veces peor. Un promedio esconde a la persona
para la que el sistema funciona peor, y con una muestra chica eso es justo lo que hay que poder
mirar (riesgo R-2).

A diferencia del generador de juguete, aquí el pico llega **cuando tiene que llegar**: el
modelo UVA/Padova incluye vaciado gástrico y absorción intestinal. El objetivo «cuándo» es el
que más gana al cambiar de origen de datos.

## Core ML, y las dos cosas que la app tiene que respetar

Los dos artefactos están convertidos y la traducción es **fiel**: contra XGBoost la diferencia
máxima en 2 000 filas es de 0.000214 mg/dL en M-1 y 0.000229 en M-2, con una tolerancia de
0.001. Eso descarta el error que no se vería de otra forma, porque un árbol traducido al revés
no falla: predice mal.

La comprobación saca además dos requisitos para `glucy-ios`, y los dos son del tipo que se
descubre en producción si no se escriben ahora:

1. **Las características van como `Float` de 32 bits, no como `Double`.** Los umbrales del
   árbol son de 32 bits y XGBoost compara contra ellos con esa misma precisión. Mandando doble
   precisión la predicción se desvía hasta **1.43 mg/dL** en M-1 (449 de 2 000 filas) y hasta
   **3.46 mg/dL** en M-2 (662 filas). No revienta nada: da otro número.
2. **Hay que comprobar en el iPhone físico que Core ML respeta la marca de valor que falta.**
   De los 4 415 cortes del bosque de M-1, **472 mandan el faltante al hijo verdadero**. Si el
   teléfono compara el `NaN` como cualquier número, la predicción se desvía hasta **21.6 mg/dL**
   en M-1 y **36.5** en M-2. Esto no se puede comprobar aquí y queda como prueba de dispositivo.

El informe de cada conversión queda junto al artefacto, en
`artifacts/m1-v0.1.0-simulador-coreml.json`, con los dos números y el entorno en que se midió.

## El cuello de botella que apareció de paso

Convertir a Core ML no arrancaba: veinte minutos sin imprimir nada. No era la conversión, era
armar las 2 000 filas de comprobación. `_mas_cercana` recorría la serie entera y se llama seis
veces por fila sobre una historia que crece hasta las 8 064 lecturas, así que el coste era
cuadrático: **68 segundos por sujeto**, veinte minutos por los dieciocho, tres veces por tanda.

Con búsqueda binaria son **1.3 segundos por sujeto**, 56 veces menos, y las filas salen
idénticas valor por valor con los `NaN` incluidos. Está aparte, en su propio PR, porque no es
un resultado de este paso.

Para el reporte, el número de ahora medido de punta a punta: **M-1 con las cinco líneas base, la
recurrente incluida, tarda 8 minutos 53 segundos** en esta máquina de cuatro núcleos. Antes, las
dos corridas con cuatro líneas base cada una tardaron dos horas y veinticuatro minutos juntas.
De esa diferencia, lo que está medido por separado es la parte de armar las filas: veinte
minutos por corrida que pasan a veinticinco segundos. El resto se explica por haber corrido M-1
y M-2 a la vez en cuatro núcleos, con XGBoost y el bosque pidiendo los cuatro cada uno, pero
**eso no se midió aparte y el reporte no debería atribuirlo con precisión**.

Que las dos corridas nuevas dan exactamente los mismos números que las viejas, con el modelo
idéntico byte por byte, es lo que confirma que el arreglo cambió el tiempo y nada más.

## Lo que este paso no hace

- **No reemplaza a OhioT1DM.** El manifiesto sale con `origen_datos: "simulador"` y
  `se_puede_reportar` sigue en falso, que solo lo pone verdadero `ohiot1dm`. Las métricas del
  simulador **van al documento final diciendo que son del simulador**, nunca sin decirlo.
- **No cumple las dos metas clínicas.** Ni M-1 ni M-2 llegan al 85 % de sensibilidad con 15 %
  o menos de falsas alarmas, en ningún umbral. El paso deja la comparación bien hecha y el
  número real; no deja un modelo aceptable.
- **No busca hiperparámetros.** El aviso del guion lo dice en el orden correcto: antes de tocar
  hiperparámetros, revisar las características. Con la comparación de arriba mal hecha, ajustar
  habría sido ajustar hacia un número que no medía lo que parecía medir.
- **No conecta Core ML en la app.** El `.mlmodel` existe y está comprobado, pero `glucy-ios`
  no tiene todavía la inferencia en el dispositivo: su pantalla de inicio sigue diciendo «Aún
  no puedo proyectar». Convertir el modelo no hace que la app prediga, y eso es trabajo aparte.
- **No prueba el faltante en el iPhone.** Es la única comprobación de la conversión que no se
  puede hacer en esta máquina, y la que más mg/dL mueve si falla.

## Órdenes para comprobar este paso

```bash
pip install "setuptools<81" simglucose
python -m src.datos.simulador --dias 28 --salida data/simulador
python -m src.train.train_m1 --datos simulador
python -m src.train.train_m2 --datos simulador
python -m src.train.train_m3 --datos simulador
python -m src.export.to_coreml --artefacto artifacts/m1-v0.1.0-simulador.json \
  --datos simulador
python -m pytest tests/test_simulador.py tests/test_umbral_de_aviso.py -q
```
