"""M-3: el **pico después de una comida** y **a los cuántos minutos**, en modo sin sensor.

Es el modelo del usuario que no tiene sensor, y por eso usa el juego disperso: la última
lectura y su antigüedad, los carbohidratos y la dosis de la comida, la hora y el contexto. **Sin
rezagos ni derivadas**, porque tres pinchazos separados por horas no son una serie y proyectar
con ellos daría apariencia de predicción a un dato que no la sostiene.

Un modelo aparte y no M-1 degradado (RF-27). La razón es la misma de siempre en este proyecto:
un modelo que dependa de rezagos entregaría basura con apariencia de predicción cuando esos
rezagos no existen.

**Son dos objetivos, y se miden distinto:**

- El **valor** del pico, en mg/dL. Meta: error ≤ 25 mg/dL. Se mide con las métricas clínicas.
- **Cuándo** ocurre, en minutos. Se mide solo con RMSE y MAE: la rejilla de Clarke clasifica el
  error de una glucosa por su consecuencia, y «zona A» sobre una cantidad de minutos no
  significa nada.

Y lo que este modelo **no** hace, que hay que dejar escrito: no sugiere nada. Predice a dónde
va a llegar la glucosa después de lo que la persona ya comió y ya se aplicó. **No calcula ni
sugiere una dosis de insulina ni gramos a tomar** (regla 7 del proyecto), y en modo sin sensor
el sistema declara que no puede anticipar hipoglucemias (RF-10b).

```bash
python -m src.train.train_m3
python -m src.train.train_m3 --datos sintetico
```
"""

from src.features.ventanas import filas_pico
from src.train.comun import argumentos, registros
from src.train.entrenar import evaluar_con_loso_simple
from src.train.modelos import XGBoost

JUEGO = "disperso"
VENTANA_MIN = 180
META_ERROR_PICO_MG_DL = 25.0


def main() -> None:
    p = argumentos(__doc__.splitlines()[0])
    p.add_argument(
        "--ventana",
        type=int,
        default=VENTANA_MIN,
        help="minutos después de la comida en los que se busca el pico",
    )
    args = p.parse_args()

    filas = []
    for r in registros(args):
        filas.extend(
            filas_pico(
                r.lecturas,
                r.eventos.get("meal", ()),
                r.eventos.get("bolus", ()),
                ventana_min=args.ventana,
            )
        )
    if not filas:
        raise SystemExit(
            "ninguna comida registrada tiene suficientes lecturas después. Sin comidas "
            "registradas no hay M-3, y no se inventan: una subida sin comida anotada no se "
            "atribuye a una comida (D-9, RF-37)."
        )
    print(f"{len(filas)} comidas de {len({f.sujeto for f in filas})} sujetos")

    valor, valor_por_sujeto = evaluar_con_loso_simple(
        filas,
        juego=JUEGO,
        objetivo=lambda f: f.pico_mg_dl,
        construir_modelo=lambda: XGBoost(semilla=args.semilla),
        unidad="mg/dL",
    )
    cuando, _ = evaluar_con_loso_simple(
        filas,
        juego=JUEGO,
        objetivo=lambda f: f.pico_min,
        construir_modelo=lambda: XGBoost(semilla=args.semilla),
        unidad="min",
    )

    print()
    print(f"Valor del pico   n={valor.n:5}  RMSE={valor.rmse:6.2f} mg/dL  MAE={valor.mae:6.2f}")
    print(f"Cuándo ocurre    n={cuando.n:5}  RMSE={cuando.rmse:6.2f} min    MAE={cuando.mae:6.2f}")
    print()
    cumple = valor.mae <= META_ERROR_PICO_MG_DL
    print(
        f"Meta del error del pico (<= {META_ERROR_PICO_MG_DL:.0f} mg/dL de MAE): "
        f"{'se cumple' if cumple else 'NO se cumple'}"
    )

    # Por sujeto, siempre: con doce personas el promedio esconde a aquella para la que el
    # sistema funciona peor, y en modo sin sensor esa es justo la persona con menos recursos
    # (riesgo R-2).
    print()
    print("Error del valor del pico por persona:")
    for sujeto, r in valor_por_sujeto.items():
        print(f"  {sujeto:8} n={r.n:4}  MAE={r.mae:6.2f} mg/dL")

    if args.datos != "ohiot1dm":
        print()
        print(
            f"AVISO: los datos son '{args.datos}'. Estas métricas NO se pueden reportar. En las "
            "series de juguete el pico llega a los pocos minutos de la comida, no a los 60-90 "
            "que tarda de verdad, así que el objetivo «cuándo» es especialmente irreal."
        )


if __name__ == "__main__":
    main()
