# Paso 1 de la fase 3 — evaluación y características

Fecha: 29 de octubre de 2026 · Rama: `feature/evaluacion-y-caracteristicas`

---

## Por qué este paso va primero, antes de cualquier modelo

El conjunto OhioT1DM todavía no llega. La tentación es esperarlo y entonces empezar, pero
**casi todo lo que decide si el resultado de la fase 3 vale algo no necesita los datos**:

- La rejilla de Clarke, que es la métrica principal del proyecto (≥ 95 % en A+B).
- Las métricas y sus metas.
- Las líneas base contra las que hay que ganar.
- Los nombres y el orden de las características, que tienen que ser idénticos a los del
  backend.
- Las reglas de partición, que son lo que más fácilmente invalida un resultado entero.

Todo eso se escribe, se prueba y se cierra ahora. Cuando lleguen los datos, el trabajo es
entrenar y medir, no decidir cómo se mide. Y medir con las reglas ya escritas evita el error
clásico: ajustar el criterio hasta que el modelo lo pase.

## Antes y después

**Antes:** el repositorio tenía el andamiaje de la fase 0 —`CLAUDE.md`, licencia, la
estructura de carpetas vacía, la integración continua— y ni una línea de código. `pytest` no
tenía nada que correr y la propia CI lo decía: «Todavía no hay pruebas en este repositorio».

**Después:** seis módulos con 107 pruebas pasando y la CI en verde. La rejilla de Clarke, las
métricas con sus metas, las dos líneas base que no necesitan entrenamiento, los dos juegos de
características con los nombres del backend, las reglas de partición y el lector de los XML de
OhioT1DM. Nada de esto necesita el conjunto de datos para correr, y todo está probado.

```
src/
  eval/
    clarke.py      rejilla de Clarke, zonas A-E
    metrics.py     RMSE, MAE, MARD, sensibilidad, falsas alarmas, y las metas
    baselines.py   persistencia y extrapolación lineal
    particion.py   por sujeto, LOSO, corte temporal, detección de fuga
  features/
    features.py    los dos juegos de RF-24, copiados del backend
  ingest/
    ohiot1dm.py    lectura de los XML
tests/             107 pruebas
requirements.txt   dependencias del entrenamiento (la CI no las instala, a propósito)
```

## Los cuatro hallazgos de este paso

### 1. La rejilla de Clarke no está implementada igual en ninguna parte

`CLAUDE.md` de este repositorio pedía cotejar la implementación de la rejilla «contra el
artículo original de Clarke et al. (1987) antes de reportar métricas». Se hizo lo que se podía
hacer: comparar `zona_clarke` de `referencia/reglas_clinicas.py` del backend contra el puerto
de Python de la implementación de MATLAB File Exchange, **par por par sobre las 337 561
combinaciones de enteros entre 20 y 600 mg/dL**.

**No coinciden en 29 690 pares, en tres familias, y no todas van en la misma dirección:**

| # | Pares | Con un real… y un predicho… | El backend dice | El puerto público dice | Ejemplo |
| --- | --- | --- | --- | --- | --- |
| 1 | 28 105 | real 181–290, predicho ≥ real + 110 | B | **C** | real 200 → 320 |
| 2 | 361 | real ≥ 240, predicho exactamente 180 | B | **D** | real 300 → 180 |
| 3 | 1 224 | real 59–70, predicho 71–180 | **D** | B | real 60 → 120 |

Qué es cada una:

1. El backend acota la zona C por arriba en 180 mg/dL en lugar de 290. Un real de 200 predicho
   como 320 invita a sobrecorregir, y eso es C, no un error benigno.
2. El backend compara con `70 < p < 180` donde el puerto usa `70 ≤ p ≤ 180`. Un `<` donde va
   un `<=`.
3. **Aquí el indulgente es el puerto público.** Su condición para la zona D baja pide
   `p ≥ (7/5)·real + 110`, que con un real de 60 son 194; como arriba de 180 ya manda la zona
   E, esa rama nunca se activa y un real de 60 predicho como 120 le sale B. Pero eso es
   literalmente «no detectar una hipoglucemia», que es la definición de la zona D y **el evento
   central de Glucy**. La fórmula del puerto parece una transcripción equivocada, y aquí el
   backend tiene razón.

Lo que importa de las tres juntas: **B cuenta para el A+B que el proyecto promete en ≥ 95 %**,
así que cada vez que una formulación llama B a lo que la otra llama C o D, infla precisamente
la cifra que decide si el modelo sirve.

Decisión: **este repositorio toma en las tres la lectura más severa** —C en la primera, D en la
segunda, D en la tercera—, así que difiere del backend en las dos primeras y del puerto público
en la tercera. Los tres casos concretos están escritos como pruebas en
`tests/test_clarke.py`, de modo que volver a cualquiera de las otras dos formulaciones hace
fallar una prueba en lugar de mover la métrica en silencio.

El criterio es sencillo: si al cotejar contra el artículo original resultara que la indulgente
es la correcta, el proyecto habría reportado una métrica **peor** que la real, y ese es el error
que se puede defender. Al contrario, no. El artículo de 1987 sigue siendo el árbitro final y no
está accesible sin suscripción; cuando se pueda consultar, se coteja. Mientras tanto, **el
documento final tiene que decir qué formulación se usó**, no solo «rejilla de Clarke».

**Queda pendiente**: corregir las dos primeras en `glucy-backend` para que los dos repositorios
den el mismo número. Va en su propio pull request, porque toca el archivo de referencia con 30
pruebas.

### 2. El esquema de los XML de OhioT1DM no está publicado

El artículo del conjunto describe los campos **en prosa y con espacios** («glucose level»,
«finger stick»), y no da el nombre del elemento raíz, ni los nombres de los atributos, ni el
formato de las marcas de tiempo. El conjunto está bajo solicitud, así que no hay ningún
archivo público con el que verificar.

Así que `SECCIONES` y `FORMATOS_DE_FECHA` de `src/ingest/ohiot1dm.py` son **una
reconstrucción**, y eso está escrito en el encabezado del módulo. Lo que sí se hizo es
encerrarla en una sola tabla de veinte líneas: cuando lleguen los archivos, cuadrar el esquema
es editar esa tabla y nada más. Todo lo que viene después trabaja sobre `Lectura` y `Evento`,
que no cambian.

Y hay una herramienta para el día que lleguen:

```python
from src.ingest.ohiot1dm import esquema_encontrado

print(esquema_encontrado("data/ohiot1dm/2020/train/559-ws-training.xml"))
```

Informa qué trae el archivo sin interpretarlo: la raíz, sus atributos, cada sección con cuántos
eventos tiene y qué atributos trae el primero, y qué no cuadra con la tabla en las dos
direcciones. **Es lo primero que hay que correr contra el primer archivo real.**

El formato de fecha merece una nota: `07-12-2021` es el 7 de diciembre, y `%d-%m-%Y` y
`%m-%d-%Y` aceptan los dos la misma cadena cuando el día es 12 o menos. Elegir mal desplazaría
medio conjunto de fecha sin que nada falle, así que cuál se eligió está fijado en una prueba.

### 3. Las características del backend se copian, no se derivan

`app/services/pipeline/caracteristicas.py` de `glucy-backend` es la autoridad y lo dice en su
propio encabezado. `src/features/features.py` copia las dos tuplas literalmente, y
`tests/test_caracteristicas.py` las fija carácter por carácter.

Parece exagerado hasta que se piensa qué pasa si no coinciden: el artefacto recibe un vector de
números en un orden, y si aquí se ordenan distinto que en el pipeline que lo alimenta en
producción, la predicción se calcula con las columnas cambiadas de lugar. Salen números
**plausibles y equivocados**. El modelo no falla, la API responde 200, y nadie se entera hasta
que las métricas de producción no se parecen a las del cuaderno. Los dos repositorios no se
pueden importar entre sí, así que la única red posible es que el cambio sea imposible de hacer
sin ver fallar una prueba.

Tres reglas que vienen con las listas y también están probadas:

- **Lo que falta es `NaN`, nunca cero.** Un rezago de 60 minutos que no existe porque la serie
  empieza ahí no es una glucosa de cero. XGBoost trata el `NaN` como información —aprende para
  qué lado mandar la fila— y el cero como un valor real que arrastra el promedio.
- **Sin `actividad` ni `sueno` en el juego denso** (C-10, C-11), aunque OhioT1DM sí traiga
  ejercicio y sueño. El teléfono no las captura y el servidor no tiene endpoint para el
  contexto, así que en producción siempre faltarían. Entrenar con una variable que nunca llega
  es peor que no tenerla: el modelo le da peso y luego no lo recibe. Es la tentación obvia al
  tener el conjunto delante y por eso hay una prueba que lo impide.
- **El peso no entra.** El conjunto lo trae como relleno —99 para todos— y dentro de una misma
  persona es constante en un horizonte de 30 minutos.

### 4. La partición se escribió como código con errores, no como una línea en un script

`train_test_split(shuffle=True)` de scikit-learn, que es lo primero que uno escribe, está mal
por dos motivos a la vez:

- Reparte al azar filas de la misma persona, así que el modelo la memoriza: aprende «esta
  fisiología a esta hora suele estar en 140» en lugar de aprender a predecir. Es el error más
  común en predicción de glucosa y el que invalida un resultado ante cualquiera que sepa
  leerlo.
- Usa datos posteriores para predecir anteriores, que es fuga de información y no existe en
  producción: a las 14:00 el teléfono no tiene las lecturas de las 15:00.

De ahí que `src/eval/particion.py` exista. Lo que hace y lo que no deja hacer:

- `partir_por_sujeto` reparte por persona y **revienta si un sujeto de prueba no está en los
  datos**, en lugar de devolver un conjunto de prueba más chico del que dice el documento.
- El propio `Reparto` revienta si un sujeto aparece en los dos lados, así que la comprobación
  cubre también un reparto armado a mano en un cuaderno.
- `pliegues_loso` devuelve una **lista y no un generador**: los pliegues se recorren una vez
  por modelo y una vez por línea base, y un generador agotado daría cero pliegues en la segunda
  pasada sin que nada falle.
- `partir_en_el_tiempo` corta **por sujeto y no sobre el conjunto entero**: los sujetos no
  comparten calendario, cada uno aportó sus ocho semanas cuando le tocó, así que un corte por
  fecha global dejaría a unos enteros en entrenamiento y a otros enteros en prueba.
- `hay_fuga_temporal` es una aserción sobre el resultado, para correr **después** de repartir:
  la fuga no se ve mirando el código que la causó.

## Otras dos decisiones

**Las metas son 85 % de sensibilidad y 15 % de falsas alarmas.** El documento de la fase F3 en
Drive trae 80 % y 20 %; es la errata conocida. Los valores correctos están en
`src/eval/metrics.py` como constantes y hay una prueba que los fija, para que nadie los
«corrija» hacia el documento equivocado.

**La sensibilidad a hipoglucemia revienta cuando el conjunto no trae episodios reales**, en
lugar de devolver 100 %. Un conjunto de prueba sin hipoglucemias no demuestra que el modelo las
detecte, y un 100 % en esa casilla sería la cifra más engañosa de todo el reporte. `evaluar`
lo convierte en `None`, que el documento tiene que escribir como «no se midió».

## Qué no hace este paso

- **No entrena nada.** M-1, M-2 y M-3 necesitan los datos.
- **No implementa Ridge, Random Forest ni LSTM.** Las tres necesitan ajustarse; llegan con
  `src/train/`. Las dos que están son las que se calculan sobre la serie misma, y son las que
  de verdad humillan a un modelo mal hecho: la glucosa tiene mucha inercia y a 30 minutos «va a
  seguir igual» acierta notablemente.
- **No toca SHAP ni la conversión a Core ML.** Van después del primer modelo entrenado.
- **No mete ningún dato al repositorio.** `data/` sigue fuera de control de versiones: lo
  prohíbe el acuerdo de uso y el repositorio es público.

## Lo que falta para que la fase 3 avance

Una sola cosa, y no es código: **el correo de solicitud de OhioT1DM sigue sin enviarse.**

- A `razvan.bunescu@charlotte.edu`, asunto `OhioT1DM Request`.
- **Tiene que salir de la dirección de Tecmilenio**, porque el correo personal se rechaza.
- Tarda hasta una semana en procesarse.
- El borrador ya está escrito y entregado en el hilo del proyecto.

Con los datos dentro, el paso 2 es construir las características sobre la serie real y medir
M-1 contra la persistencia. Sin ellos, el 8 de noviembre es la fecha de corte para cambiar al
simulador fisiológico y decirlo así en el reporte.

## Órdenes para comprobar este paso

```bash
ruff check .      # estilo
ruff format --check .
pytest -q         # 107 pruebas, segundos, sin instalar nada más que pytest y ruff
```
