# Olist Store — Análisis de Sentimiento en Reviews

Clasificador binario de reseñas en portugués (negativa 1-2★ vs. positiva 4-5★) para detectar reclamos y priorizar atención al cliente.

## Quick path

1. Instalar deps: `uv sync` o `pip install -r requirements.txt`
2. Abrir `olist_nlp.ipynb` y correr en orden
3. Resultado esperado: SVM con accuracy ~94.9%, recall-negativo 93.3%

## Details

| Tema | Decisión |
|------|----------|
| Datos | `datasets/olist_order_reviews_dataset.csv` — 39.044 con texto, sin 3★ |
| Pipeline | Título + mensaje → TF-IDF (1-2 gramas, stopwords PT preservando negaciones) |
| Split | Temporal 80/20 (31.235 train / 7.809 test), TF-IDF fit solo en train |
| Modelo | Linear SVM elegido sobre Naive Bayes por mejor recall en negativos |
| Output | Matriz de confusión + top 15 palabras por sentimiento |

## Next step

Ver conclusión en `olist_nlp.ipynb` → sección 7.
