# Paso 4 — la quinta línea base, la recurrente

El protocolo pide cinco líneas base obligatorias: persistencia, extrapolación lineal, Ridge,
Random Forest y **LSTM**. Había cuatro. Esta es la quinta.

## Por qué se escribió ahora, si no hay datos

Porque no depende de los datos. Es código, se prueba con series de juguete y tiene que estar
lista el día que el conjunto llegue. Dejarla para entonces convertiría una tarde de trabajo en
un riesgo del calendario, y la fase 3 no tiene holgura.

Y porque el argumento entero del proyecto es «el modelo le gana a estas cinco». Una tabla con
cuatro es una tabla que hay que explicar en el reporte, y «no la escribimos» no es una
explicación.

## De dónde sale la secuencia, que es la decisión del paso

Una red recurrente quiere una secuencia; el bucle de evaluación reparte filas planas. Armar un
conjunto aparte para la LSTM habría sido comparar dos modelos sobre dos conjuntos distintos,
que es exactamente la trampa que el diseño de `evaluar_con_loso` existe para evitar.

La secuencia se arma **con las columnas que la fila ya trae**:

```
rezago_60 → rezago_30 → rezago_15 → rezago_10 → rezago_5 → mg_dl
```

Son seis instantes ordenados de la misma glucosa; las características de rezago **son** la
serie. El resto —COB, IOB, hora cíclica, día de la semana— se repite en cada paso como
contexto.

Así la red ve **la misma información** que XGBoost, sobre **las mismas filas** y con **la misma
partición**. Lo que se mide es la arquitectura, que es lo que se quería medir.

**Lo que hay que decir en el reporte:** los seis pasos no están igualmente espaciados. Van 30,
15, 5, 5 y 5 minutos. Una LSTM no asume espaciado uniforme —no es un filtro—, así que no es un
error, pero dejar creer que son seis muestras seguidas del sensor sí lo sería.

## Por qué está entrenada con cuidado y no de cualquier manera

Una línea base mal entrenada **adula al modelo del proyecto**. Es la misma razón por la que la
extrapolación lineal mide su pendiente sobre 30 minutos y no sobre 5, y es la dirección en la
que no conviene equivocarse.

Por eso lleva lo que una red necesita para no salir artificialmente mal:

| Decisión | Por qué |
| --- | --- |
| Estandarizar entradas y objetivo | sin esto una red con glucosas de 40 a 400 aprende despacio o no aprende |
| La mediana y la desviación salen **solo del entrenamiento** | calcularlas sobre todo el conjunto filtra información de la prueba |
| Imputar los `NaN` con la mediana | una red no los acepta. **Es parte de esta línea base**, igual que en Ridge: si sale mal, parte puede ser la mediana |
| Adam, 60 épocas máximas | suficiente para una red de una capa y 32 unidades |
| Paro temprano con paciencia de 8 | y se devuelven los pesos de la **mejor** época, no los de la última |
| El corte de validación es **temporal**, el último 20 % en orden | barajar la serie y validar sobre el pasado es fuga de información: daría una línea base irrealmente buena y un paro en la época equivocada |
| Semilla fija en torch y en numpy | un resultado que no se puede repetir no sirve para un trabajo que se va a evaluar |

`epocas_usadas` entra a los hiperparámetros del manifiesto: «60 máximas» no dice nada si paró
en la 12.

## Cómo se corre

Va **encendida por omisión** en los tres guiones. `--sin-lstm` la salta, y está para iterar, no
para reportar: quien lo escribe sabe lo que se está saltando.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m src.train.train_m1                 # con las cinco líneas base
python -m src.train.train_m1 --sin-lstm      # rápido, tabla incompleta
```

Con el juego disperso **se niega a construirse**, con un mensaje que lo dice: sobre un solo
punto no hay serie que recorrer, y una LSTM ahí no es una LSTM. Es decir, M-3 se compara contra
cuatro líneas base y eso es correcto, no una omisión.

## Pruebas

Siete, en `tests/test_modelos.py`. Tres corren sin PyTorch —la forma de la secuencia, la
imputación y la negativa con el juego disperso son aritmética, no red— y cuatro se saltan solas
en la integración continua, que no instala nada pesado.

La que importa es `test_la_lstm_aprende_una_tendencia_y_le_gana_a_la_persistencia`: sobre
series con tendencia clara, la recurrente tiene que quedar **por debajo** del error de repetir
el valor de ahora. Sin esa prueba, una red mal cableada pasaría por línea base y haría quedar
bien a XGBoost sin que nadie lo notara.

## Lo que esta línea base todavía no dice

Si la LSTM se sostiene con doce sujetos. Eso se sabe midiendo, con los datos delante, y es un
resultado del reporte. Lo que ya no se puede decir es que no se intentó.
