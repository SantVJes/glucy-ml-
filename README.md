# glucy-ml

> **Glucy — «Tu glucosa, media hora antes».**
> Sistema que registra glucosa por tres vías intercambiables, la predice a 30 y 60 minutos con
> un modelo de aprendizaje automático y explica cada predicción en español llano.
> Proyecto académico de la Universidad Tecmilenio, 8.º semestre, materia Desarrollo de
> Proyectos de Aplicaciones Móviles. Autor: Jesús Santiago Velasco.

**Qué hace este repositorio:** el entrenamiento, la validación y los artefactos del modelo predictivo. Python con XGBoost, scikit-learn y SHAP. Produce el modelo que la aplicación ejecuta dentro del teléfono con Core ML.

Los otros dos: [glucy-ios](https://github.com/SantVJes/glucy-ios) ·
[glucy-backend](https://github.com/SantVJes/glucy-backend) ·
[glucy-ml](https://github.com/SantVJes/glucy-ml-)


## Cómo levantarlo

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env          # y llenarlo; .env nunca se sube
python -m src.train.train_m1  # a partir de la fase 3
```

## Los datos no están aquí

**El conjunto OhioT1DM no se versiona**: el acuerdo de uso prohíbe redistribuirlo y este
repositorio es público. Cómo conseguirlo y dónde ponerlo: [`data/README.md`](data/README.md).

## Variables de entorno

Están en [`.env.example`](.env.example). Incluyen la semilla, porque **toda corrida deja su
semilla, sus hiperparámetros y sus métricas junto al artefacto**: un resultado que no se puede
repetir no sirve para un trabajo que se va a evaluar.

## Cómo correr las pruebas

```bash
pytest -q
ruff check .
```

## Estructura de carpetas

| Carpeta | Qué contiene |
| --- | --- |
| `data/` | el conjunto de datos, **fuera del control de versiones** |
| `src/ingest/` | lectura de los XML de OhioT1DM |
| `src/features/` | los dos juegos de características, denso y disperso |
| `src/train/` | M-1 a 30 min, M-2 a 60 min y M-3 del pico postprandial |
| `src/eval/` | rejilla de Clarke, métricas y líneas base |
| `src/export/` | conversión a Core ML |
| `notebooks/` | cuaderno de evaluación |
| `artifacts/` | modelos versionados con su JSON de métricas |

## Metas del modelo

Rejilla de Clarke zonas A+B **≥ 95 %** · sensibilidad a hipoglucemia **≥ 85 %** · tasa de
falsas alarmas **≤ 15 %** · RMSE a 30 min ≤ 20 mg/dL · MAE ≤ 15 mg/dL · MARD ≤ 12 % · error
del pico postprandial ≤ 25 mg/dL.

Un modelo que no le gane a la línea base de persistencia —«dentro de 30 minutos será igual que
ahora»— no justifica su existencia.

## Sobre el sesgo, dicho con el número en la mano

El conjunto de entrenamiento son **doce personas** con diabetes tipo 1, ocho semanas cada una.
Los resultados **no son generalizables a población abierta**, y así se reportan.

## Estado

Fase 0 terminada: repositorio con su estructura, su `CLAUDE.md`, su integración continua y su
documentación base. **La construcción empieza en la fase 1.** El plan completo, con las siete
fases repartidas hasta finales de noviembre de 2026, está en la base de documentación del
proyecto.

## Aviso

**Glucy no es un dispositivo médico certificado.** El sistema informa; nunca prescribe ni
administra, y **jamás calcula ni sugiere una dosis de insulina**. No debe usarse como única
fuente para una decisión terapéutica. Ante cualquier síntoma hay que medir, prediga lo que
prediga la aplicación.

## Licencia y cómo citar

Licencia MIT, ver [LICENSE](LICENSE). Todo el stack del proyecto es software libre y gratuito.

> Santiago Velasco, J. (2026). *Glucy: aplicación iOS de monitoreo y predicción de glucosa con
> aprendizaje automático e inteligencia artificial generativa* [software]. Universidad
> Tecmilenio.
