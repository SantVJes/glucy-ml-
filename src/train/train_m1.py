"""M-1: la glucosa dentro de **30 minutos**, en modo sensor.

Es el modelo principal del proyecto y el primero que hay que medir, porque es el que sostiene
la promesa del eslogan: «tu glucosa, media hora antes». Las metas a 30 minutos son RMSE
≤ 20 mg/dL y MAE ≤ 15 mg/dL, y por encima de las dos está el criterio que decide: **≥ 95 % de
las predicciones en las zonas A+B de la rejilla de Clarke**.

Un modelo independiente por horizonte y no uno que se degrada (RF-26): a 60 minutos la
incertidumbre es otra, y un solo modelo estirado reparte mal su capacidad entre los dos plazos.

```bash
python -m src.train.train_m1                      # con el conjunto en data/ohiot1dm
python -m src.train.train_m1 --datos sintetico    # para probar el guion sin datos
```
"""

from src.train.comun import argumentos, correr

HORIZONTE_MIN = 30


def main() -> None:
    args = argumentos(__doc__.splitlines()[0]).parse_args()
    correr(args, modelo="m1", horizonte_min=HORIZONTE_MIN)


if __name__ == "__main__":
    main()
