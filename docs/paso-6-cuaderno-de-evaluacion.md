# Paso 6 — el cuaderno de evaluación y las cuatro figuras

La estructura documentada de este repositorio lleva desde el principio una carpeta
`notebooks/` con «el cuaderno de evaluación, con la rejilla de Clarke dibujada», y estaba
vacía. Es un entregable del reporte, no un extra.

## Lo que se construyó

```
src/eval/dibujos.py        las cuatro figuras
src/eval/cuaderno.py       lo que hay que calcular antes de dibujar, y la orden que lo deja todo
notebooks/evaluacion.ipynb el cuaderno, que solo llama y enseña
tests/test_dibujos.py      16 pruebas
tests/test_cuaderno.py      8 pruebas, una de ellas corre el cuaderno entero
```

```bash
python -m src.eval.cuaderno --salida artifacts/figuras
```

## La decisión principal: la rejilla se dibuja con el clasificador, no con rectas

La forma natural de dibujar una rejilla de Clarke es trazar a mano las rectas que separan las
zonas. Es también la forma de que el dibujo y el número dejen de coincidir sin que nadie lo
note: `src/eval/clarke.py` cambió **dos veces** en este proyecto, y la última movió 19 900 pares
de B a C. Unas rectas escritas aparte habrían seguido diciendo lo de antes, y una figura que
contradice a la métrica que la acompaña es peor que ninguna, porque se publica.

Así que el fondo se pinta **evaluando `zona()` sobre una malla de 1 mg/dL**. El dibujo no puede
discrepar del clasificador porque es el clasificador. Cuesta unas 337 000 llamadas en el peor
caso, que tardan un segundo.

### Y una falla que salió de ahí

Las letras de las zonas se ponían en el **centro de masa** de cada zona. La zona E son dos
esquinas opuestas, así que su centro de masa cae en medio del dibujo, **encima de la zona A**. La
primera rejilla que se dibujó rotulaba «E» sobre la zona A y «D» sobre la B.

Se arregla en dos pasos sobre la misma malla: se parten las zonas en trozos conexos —la E son
dos, la D son tres— y en cada trozo se busca el punto más lejano de cualquier frontera, con un
recorrido en anchura desde todas las fronteras a la vez. Ahí cabe la letra sin tocar la zona
vecina. Los trozos que ocupan menos de la centésima parte del dibujo no se rotulan: la letra
taparía más de lo que explica.

`test_01_cada_letra_cae_dentro_de_su_propia_zona` comprueba exactamente eso, llamando a
`zona()` en el punto donde quedó cada letra.

## Las otras tres figuras, y por qué cada una

| Figura | Qué enseña que un número no enseña |
| --- | --- |
| **Lo medido contra lo predicho** | si el modelo **llega tarde**. Un RMSE de 15 mg/dL puede ser una curva que sigue bien la forma o una que la sigue con diez minutos de retraso, y la segunda no sirve para anticipar nada. |
| **Contra las líneas base** | «un modelo que no le gana a la persistencia no justifica su existencia», de un vistazo: lo que queda a la derecha de la línea está perdiendo. |
| **Error por origen del dato** | el riesgo **R-2**: si el sistema funciona peor precisamente para quien no tiene sensor, que es el usuario con menos recursos. Un promedio único lo esconde. |

Las tres respetan dos reglas de interfaz del proyecto, que aquí valen igual que en la app porque
quien lee el reporte puede estar igual de cansado:

- **Lo medido continuo y lo predicho punteado.** Es la regla que el simulador de la fase 4 rompió
  sin que una revisión de código lo viera, así que aquí tiene prueba:
  `test_10_lo_medido_sale_continuo_y_lo_predicho_punteado`.
- **El color nunca es el único portador.** Las zonas van de claro a oscuro **y** rotuladas; el
  modelo del proyecto se distingue por el tono **y** por un asterisco en su etiqueta. Las figuras
  se entienden impresas en gris.

## El cuaderno corre hoy, sin datos y sin modelo

No hay OhioT1DM ni modelo entrenado, y faltan semanas. Un cuaderno que no se pueda abrir hasta
entonces es un cuaderno que se escribe de prisa al final.

- Si `data/` está vacía, cae al generador de juguete, **y lo dice en cada figura y en la salida**:
  una corrida sintética no es un resultado (riesgo R-8).
- Lo que se evalúa mientras tanto es la **persistencia**. No es relleno: es el listón que el
  protocolo pone, así que es exactamente la figura contra la que se comparará M-1. Tener el listón
  puesto antes de medir es el orden correcto.
- Con una sola barra, la figura de las líneas base dice «todavía no hay modelo» en lugar de
  «todos le ganan a la persistencia», que con una sola barra, que es ella, sería una figura que se
  felicita sola.

## Dos cosas que no se versionan, y por el mismo motivo

- **Las salidas del cuaderno.** Una salida guardada se queda con los números de la corrida en que
  se guardó y los sigue enseñando cuando el clasificador cambia, sin avisar.
- **`artifacts/figuras/`**, que ya está en `.gitignore`. Se regeneran con una orden.

## Por qué la lógica no está en el cuaderno

Un cuaderno es el único archivo del repositorio que nadie compila, nadie prueba y el estilo no
revisa. Se rompe cuando cambia una firma y se descubre al abrirlo, que con suerte es la semana de
la entrega. Así que lo que decide algo está en los dos módulos, y el cuaderno llama y enseña.

Y para que ni siquiera eso se pudra, `test_30_el_cuaderno_corre_entero` **ejecuta sus celdas de
código**, en orden y en un mismo espacio de nombres, en una carpeta temporal. No comprueba qué se
dibuja —para eso están las otras pruebas—, solo que corre.

## Lo que falta

- El desglose por origen necesita predicciones sobre lecturas manuales y por foto, y eso sale del
  pipeline con datos de verdad. Hoy la figura se arma con el único origen que hay: sirve para ver
  la forma que tendrá, no para concluir nada.
- Las cinco líneas base del protocolo se dibujan en cuanto haya una corrida de entrenamiento que
  las produzca; el diccionario ya las acepta.
