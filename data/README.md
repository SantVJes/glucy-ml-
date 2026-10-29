# data/

**El conjunto de datos no se versiona.** El acuerdo de uso de OhioT1DM prohíbe
redistribuirlo y este repositorio es público, así que `data/` está en `.gitignore` salvo
este archivo.

## Cómo conseguirlo

OhioT1DM: 12 personas con diabetes tipo 1, 8 semanas de datos cada una. Es gratuito para uso
académico. **No hay formulario ni acuerdo que firmar: es un correo** (comprobado en la página
oficial el 13 de octubre de 2026; los documentos del proyecto decían otra cosa).

https://webpages.charlotte.edu/rbunescu/data/ohiot1dm/OhioT1DM-dataset.html

- A `razvan.bunescu@charlotte.edu`, asunto exacto `OhioT1DM Request`. Razvan Bunescu ya no
  está en Ohio University sino en UNC Charlotte, y ahí vive la página; buscar
  «smarthealth.cs.ohio.edu» lleva a la versión vieja.
- **Desde el correo institucional**: la página dice que los personales no se aceptan.
- El cuerpo lleva nombre, puesto, correo institucional y el nombre de la institución con su
  dirección completa.
- Tarda **hasta una semana**. Llega un enlace a una copia cifrada y la contraseña aparte.

El correo se envió el **29 de octubre de 2026**.

Marling, C., & Bunescu, R. (2020). *The OhioT1DM dataset for blood glucose level prediction:
Update 2020.* CEUR Workshop Proceedings, 2675, 71–74.

## Dónde va una vez descargado

```
data/
  ohiot1dm/
    2018/  train/  test/
    2020/  train/  test/
```

## Si no llega a tiempo

El respaldo es el **simulador fisiológico UVA/Padova (2008) por simglucose**, con menos
validez, **y se declara como tal en el reporte** en lugar de afirmar una validación que no
ocurrió. Está escrito y corriendo: `src/datos/simulador.py`.

```bash
pip install "setuptools<81" simglucose     # el tope no es decorativo: ver requirements.txt
python -m src.datos.simulador --dias 28 --salida data/simulador
python -m src.train.train_m1 --datos simulador
```

Deja un JSON por sujeto en `data/simulador/`, que tampoco se versiona: se regenera de la
semilla. Los tres entrenamientos **leen de disco y no simulan**, para que M-1, M-2 y M-3 estén
medidos sobre exactamente el mismo conjunto.

**Son 18 de los 30 sujetos de simglucose**, no los 30: los diez niños tienen entre 7 y 12
años y dos adolescentes tienen 14, por debajo de la edad mínima de 15 del perfil de Glucy.
La lista no está escrita a mano, `tests/test_simulador.py` la comprueba contra las edades del
propio simglucose en los dos sentidos.

Lo que el reporte tiene que decir además: el controlador basal-bolo del simulador **calcula el
bolo con el conteo exacto de carbohidratos de la comida**, así que la variabilidad es menor que
la de una persona real, que estima a ojo. Medido sobre diez días de un adulto: 4.5 % del tiempo
por debajo de 70 mg/dL y un mínimo de 51 mg/dL, es decir que sí hay hipoglucemias que predecir,
pero menos desordenadas que las de verdad.
