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

La primera corrida de M-1 sobre el simulador, con 145 062 filas y 18 pliegues:

| | RMSE | MAE | MARD | A+B | Sens. hipo | Falsas |
| --- | --- | --- | --- | --- | --- | --- |
| M-1 | **14.06** | **10.52** | **8.75 %** | 98.96 % | 59.2 % | 20.2 % |
| persistencia | 19.64 | 14.08 | 11.15 % | 99.08 % | **67.8 %** | 32.1 % |
| extrapolación lineal | 24.14 | 17.84 | 14.38 % | 99.16 % | 75.3 % | 58.4 % |
| ridge | 14.88 | 11.13 | 9.36 % | 98.46 % | 45.9 % | 17.4 % |
| bosque | 14.24 | 10.60 | 8.77 % | 98.96 % | 59.5 % | 20.7 % |

Cumple cuatro metas de seis: RMSE ≤ 20, MAE ≤ 15, MARD ≤ 12 % y A+B ≥ 95 %. **Falla las dos
clínicas**: sensibilidad 59.2 % contra una meta de 85 %, y falsas alarmas 20.2 % contra 15 %.

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

## Lo que este paso no hace

- **No reemplaza a OhioT1DM.** El manifiesto sale con `origen_datos: "simulador"` y
  `se_puede_reportar` sigue en falso, que solo lo pone verdadero `ohiot1dm`. Las métricas del
  simulador **van al documento final diciendo que son del simulador**, nunca sin decirlo.
- **No busca hiperparámetros.** El aviso del guion lo dice en el orden correcto: antes de tocar
  hiperparámetros, revisar las características. Con la comparación de arriba mal hecha, ajustar
  habría sido ajustar hacia un número que no medía lo que parecía medir.
- **No conecta Core ML en la app.** `glucy-ios` no tiene todavía la inferencia en el
  dispositivo; su pantalla de inicio sigue diciendo «Aún no puedo proyectar». Entrenar el
  modelo no hace que la app prediga, y eso es trabajo aparte.

## Órdenes para comprobar este paso

```bash
pip install "setuptools<81" simglucose
python -m src.datos.simulador --dias 28 --salida data/simulador
python -m src.train.train_m1 --datos simulador
python -m src.train.train_m3 --datos simulador
python -m pytest tests/test_simulador.py tests/test_umbral_de_aviso.py -q
```
