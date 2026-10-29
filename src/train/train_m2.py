"""M-2: la glucosa dentro de **60 minutos**, en modo sensor.

Las mismas características que M-1 y un modelo aparte (RF-26). A una hora el error es
naturalmente mayor, así que **no se compara contra las metas de 30 minutos**: castigarlo por
eso sería castigarlo por algo que no es su culpa. Lo que sí se le exige igual es el criterio
clínico, ≥ 95 % en A+B, y que le gane a la persistencia, que a 60 minutos es un rival bastante
más débil que a 30.

```bash
python -m src.train.train_m2
python -m src.train.train_m2 --datos sintetico
```
"""

from src.train.comun import argumentos, correr

HORIZONTE_MIN = 60


def main() -> None:
    args = argumentos(__doc__.splitlines()[0]).parse_args()
    correr(args, modelo="m2", horizonte_min=HORIZONTE_MIN)


if __name__ == "__main__":
    main()
