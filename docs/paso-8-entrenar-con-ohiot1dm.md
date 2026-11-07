# Paso 8 · Entrenar con OhioT1DM

El 7 de noviembre de 2026 llegó el conjunto OhioT1DM: doce personas con diabetes tipo 1, ocho
semanas cada una. **Los datos no se suben**: viven en `data/ohiot1dm/`, que está en `.gitignore`,
porque lo prohíbe el acuerdo de uso y el repositorio es público.

## Lo que hubo que arreglar antes de entrenar

El conjunto se descomprime en `2018/` y `2020/`, cada uno con `train/` y `test/`, y cada persona
viene en dos archivos: el principio y el final de la misma serie. El lector solo miraba una
carpeta plana. Leídas por separado, la misma persona habría quedado en entrenamiento y en prueba
de la validación dejando un sujeto fuera, que es justo la fuga que esa partición evita.
`leer_conjunto` busca en subcarpetas y `unir` junta las dos mitades en una sola serie ordenada y
sin repetir.

## Resultados (`v0.3.0-ohiot1dm`, 12 pliegues dejando una persona fuera)

M-1, 30 min, 164 444 filas:

| | RMSE | MAE | MARD | A+B | Sens. hipo | Falsas |
| --- | --- | --- | --- | --- | --- | --- |
| Meta | ≤ 20 | ≤ 15 | ≤ 12 % | ≥ 95 % | ≥ 85 % | ≤ 15 % |
| **M-1** | 20.48 | 14.37 | 10.10 % | 98.09 % | 27.0 % | 26.6 % |
| persistencia | 24.86 | 17.73 | 12.14 % | 98.67 % | 52.9 % | 46.9 % |
| extrapolación | 28.85 | 19.43 | 13.39 % | 99.00 % | 80.8 % | 63.0 % |
| ridge | 21.47 | 15.11 | 10.57 % | 98.43 % | 40.8 % | 38.8 % |
| bosque | 20.58 | 14.39 | 10.09 % | 98.15 % | 30.5 % | 28.4 % |

M-2, 60 min, 162 652 filas: RMSE 33.90 (persistencia 40.44), A+B 95.64 %, sensibilidad 1.6 %,
falsas 40.6 %.

M-3, 2 024 comidas: error del valor del pico **40.45 mg/dL** (meta ≤ 25), del momento 45.28 min.
Por persona, de 34.89 a 50.75 mg/dL. No se guarda artefacto de M-3: el guion solo evalúa.

**Ninguno le gana a la persistencia** según la regla del proyecto: mejor RMSE, pero A+B 0.58
puntos peor en M-1. Igualando la tasa de falsas alarmas, M-1 (aviso < 79) caza **10.8 puntos
más** que la persistencia; M-2 (aviso < 88) caza 2.3 menos.

Peso de las características en M-1 (ganancia total): `mg_dl` 78.0 %, `rezago_5` 14.5 %,
`derivada_1` 5.2 %, `cob` 0.4 %, `iob` 0.4 %. El modelo es una persistencia refinada: lo
siguiente son mejores características de comida e insulina, no hiperparámetros.

Comparado con el simulador: RMSE de M-1 de 14.11 a 20.48, sensibilidad de 59.5 % a 27.0 %, y M-3
de 22.60 (cumplía) a 40.45. El simulador era optimista.

## Lo que falta

- **La línea base LSTM no se corrió con estos datos**: con las otras cuatro la corrida tardó
  1 h 50 min y con la LSTM no cabía. El protocolo pide las cinco y el reporte lo dice.
- Convertir `v0.3.0-ohiot1dm` a Core ML (`src/export/to_coreml.py`).

Riesgo R-2: doce personas tipo 1; los resultados no son generalizables a población abierta.
