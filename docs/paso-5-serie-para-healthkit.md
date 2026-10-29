# Paso 5 — la serie para HealthKit (plan B de la fase 5)

## El problema que resuelve

La fase 5 valida el modo sensor con **48 horas continuas**: la consulta en segundo plano
despertando a la app, los huecos medidos, los retrasos medidos, la alerta encendida y el
comportamiento cuando la persona revoca el permiso desde Salud.

Eso pide un sensor comercial, y el proyecto no tiene uno. La fecha para decidir si se consigue
es el **1 de noviembre de 2026**. El plan B documentado es escribir una serie en HealthKit desde
una utilidad de pruebas, para que la app la lea como leería la de un Dexcom o un Libre.

Este paso produce lo que esa utilidad necesita: `src/datos/para_healthkit.py` y su CSV.

## Lo que este plan B valida, y lo que no

Valida **la plomería**, que es donde están las tres trampas documentadas de HealthKit:

1. `HKObserverQuery` con entrega en segundo plano, que es lo único que no se puede probar en el
   simulador.
2. **No reescribir lo que se leyó** (RF-15b, caso P-14). La utilidad escribe desde su propio
   `HKSource`, distinto del de Glucy, así que la app ve datos ajenos y tiene que dejarlos en paz.
   Es el caso que hoy está sin verificar en `docs/release-v0.1.0.md`.
3. No caerse cuando se revoca el permiso con la app abierta (RF-07, caso P-09).

Y en una cosa valida **mejor** que un sensor real: los huecos y los retrasos se piden a
propósito en lugar de esperar a que ocurran.

**No valida el comportamiento del sensor comercial**: su cadencia real, sus recuperaciones, sus
calibraciones, lo que escribe al calibrar. **Eso tiene que decirlo el reporte (riesgo R-8)**, no
dejar creer que se validó con hardware.

## El archivo, y por qué lleva tres columnas

```
ts_medicion,ts_escritura,mg_dl
2026-01-05T00:00:00-06:00,2026-01-05T00:00:00-06:00,103.4
```

- `ts_medicion` — cuándo ocurrió la lectura. Es lo que va en el `HKQuantitySample`.
- `ts_escritura` — cuándo la utilidad debe escribirla. La utilidad ordena por esta columna y
  respeta las pausas.

Dos columnas de tiempo porque HealthKit **no entrega una serie regular de cinco minutos**:
entrega lo que la app del fabricante escribió, cuando lo escribió, y puede llegar en bloques y
con retraso (RF-06b). Una utilidad que escribiera una serie perfecta cada cinco minutos
validaría el único escenario que nunca ocurre.

## Las tres decisiones

### 1. Un hueco es una lectura que falta, no un valor nulo

Así es como falta un dato en HealthKit: no hay muestra. Un cero o un `NaN` escrito en la base
sería otra cosa y la app tendría que distinguirlo. Los largos se reparten a los dos lados de los
**15 minutos** —el umbral de interpolación (RF-23)— para que el archivo ejercite las dos ramas:
la que la app puede rellenar y la que la degrada a modo sin sensor.

### 2. Las lecturas atrasadas llegan en bloque, con una sola marca de escritura

Un sensor que vuelve a alcance no escribe sus lecturas una por una y a su hora: las escribe
**todas de golpe** en el momento de volver. Por eso todas las de un bloque comparten
`ts_escritura`, y por eso cada bloque tiene más de una lectura: una sola no probaría que la app
aguanta un lote. Dos días de serie salen con tres bloques de unas siete lecturas cada uno.

Consecuencia que la prueba fija: **las escrituras nunca van hacia atrás**. Si el archivo pidiera
escribir en el pasado, la pausa entre dos escrituras saldría negativa y la utilidad tendría que
decidir qué hacer con eso; el archivo no le pasa esa decisión.

### 3. Las marcas salen con desplazamiento, aunque el origen no lo traiga

Regla 5 del proyecto: UTC más la zona, siempre las dos. Ninguna de las dos series de origen trae
zona —OhioT1DM es hora local del sujeto sin decir cuál, y el generador de juguete tampoco la
pone—, así que aquí se les **asigna** la de México. Es una convención del archivo de pruebas, no
un dato del conjunto, y por eso está dicho en el módulo.

## Cómo se corre

```bash
# Con el generador de juguete, que es lo que hay hoy
python -m src.datos.para_healthkit --dias 2 --salida serie-para-healthkit.csv

# Con OhioT1DM, cuando llegue
python -m src.datos.para_healthkit --datos ohiot1dm --carpeta data --sujeto 559 --dias 2

# Series de control, para descartar causas si la prueba del teléfono falla
python -m src.datos.para_healthkit --sin-huecos --sin-retrasos
```

Imprime el resumen que se copia al reporte, y con datos sintéticos imprime además el aviso R-8:

```
serie.csv: 535 lecturas en 48.0 h · 16 huecos (3 de más de 15 min, el mayor de 90 min)
          · 20 lecturas que llegan tarde, en 3 bloques
AVISO: la serie es sintética. Sirve para validar la plomería de HealthKit, no el
comportamiento de un sensor comercial. El reporte tiene que decirlo (R-8).
```

La semilla es fija: dos corridas dan el mismo archivo. Una prueba de 48 horas que no se puede
repetir no sirve para comparar un arreglo contra el problema que arregla.

## Pruebas

`tests/test_para_healthkit.py`, 19 pruebas, biblioteca estándar nada más, así que corren en la
integración continua sin instalar nada. Las que sostienen el módulo:

| Prueba | Qué fija |
| --- | --- |
| 01 | hay huecos de los dos lados de los 15 minutos |
| 02 | un hueco es una lectura que falta, y ningún valor sale del rango 20–600 |
| 10 | cada bloque atrasado comparte una sola `ts_escritura` y tiene más de una lectura |
| 12 | las escrituras nunca van hacia atrás |
| 20 | la misma semilla da exactamente el mismo archivo |
| 21 | las marcas salen con zona aunque el origen no la traiga |
| 30 | el viaje redondo por el CSV no pierde la zona ni los decimales |
| 60 | los pinchazos sueltos no entran: solo la serie del sensor continuo |
| 61 | la línea de órdenes avisa del riesgo R-8 |

## Lo que falta, y está del otro lado

La utilidad que **lee** este CSV y escribe en HealthKit es código de iOS y va en `glucy-ios`,
con su propio documento de paso. Tiene que escribir desde un `HKSource` distinto del de Glucy,
o la prueba de RF-15b no prueba nada.
