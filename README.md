# Titanic – Clasificación Binaria con MLP en PyTorch

## 1. Objetivo del notebook

Este notebook tiene como objetivo **construir, entrenar y evaluar un modelo de clasificación binaria usando PyTorch**, aplicado al dataset clásico de **Titanic**, comparándolo de forma rigurosa contra un **baseline lineal (Logistic Regression)**.

El foco no es solo el resultado, sino **entender cada decisión técnica**:

* preparación de datos
* definición explícita del modelo
* función de pérdida y optimizador
* entrenamiento por batches
* evaluación con AUC
* ajuste de umbral (threshold)
* manejo de desbalance con `pos_weight`

---

## 2. Dataset

Se utiliza el dataset **Titanic** (Kaggle):

* Target: `Survived` (0 = no sobrevive, 1 = sobrevive)
* Dataset pequeño, tabular y parcialmente desbalanceado

Este dataset es ideal para:

* demostrar buenas prácticas
* entender por qué deep learning **no siempre** supera modelos lineales

---

## 3. Preparación de datos

### 3.1 Separación de variables

* Variables numéricas: escaladas
* Variables categóricas: One-Hot Encoding

Se utiliza un `ColumnTransformer` de sklearn para:

* imputar valores faltantes
* estandarizar numéricas
* codificar categóricas

Resultado:

* Matriz numérica final lista para ML

---

## 4. Split Train / Test

Se realiza un split estratificado:

* Train: 80%
* Test: 20%

La estratificación asegura que la proporción de sobrevivientes se mantenga en ambos conjuntos.

---

## 5. Baseline: Logistic Regression

Antes de usar deep learning, se entrena un **modelo base**:

* Modelo: `LogisticRegression`
* Métrica principal: **ROC AUC**

Resultados típicos:

* AUC ≈ 0.83–0.84

Este baseline sirve como **mínimo aceptable**.
Si un modelo más complejo no lo supera, no está justificado.

---

## 6. Preparación para PyTorch

### 6.1 Conversión a tensores

* Las matrices `sparse` se convierten a `numpy`
* Luego a `torch.Tensor`
* Tipo: `float32`

Se ajustan shapes para que:

* `X` tenga forma `(N, features)`
* `y` tenga forma `(N, 1)`

---

## 7. DataLoader

Se utiliza:

* `TensorDataset` para emparejar `(X, y)`
* `DataLoader` para entrenar por batches

Parámetros clave:

* `batch_size = 32`
* `shuffle = True` (solo en train)

Esto permite entrenamiento estable y eficiente.

---

## 8. Device (CPU / GPU)

Se define explícitamente el dispositivo:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Tanto el modelo como los datos se mueven al mismo `device`.

---

## 9. Modelo: MLP en PyTorch

Se define una **red neuronal multicapa (MLP)** heredando de `nn.Module`.

Arquitectura base:

* Linear → ReLU → Dropout
* Linear → ReLU → Dropout
* Linear (salida)

La salida es un **logit**, no una probabilidad.

Esto permite usar `BCEWithLogitsLoss` (más estable numéricamente).

---

## 10. Función de pérdida y optimizador

### 10.1 Loss

```python
nn.BCEWithLogitsLoss()
```

* Aplica sigmoid internamente
* Calcula binary cross-entropy

### 10.2 Optimizador

```python
torch.optim.Adam(model.parameters(), lr=1e-3)
```

Adam se encarga de ajustar los pesos usando los gradientes.

---

## 11. Loop de entrenamiento

El entrenamiento se hace explícitamente:

Para cada batch:

1. Forward
2. Cálculo de loss
3. Backpropagation
4. Optimizer step

Se entrena por **epochs**, con **early stopping** basado en AUC de validación.

---

## 12. Evaluación

### 12.1 Función `predict_proba`

Se define una función dedicada para:

* poner el modelo en modo evaluación
* desactivar gradientes
* devolver probabilidades

Esto permite calcular:

* ROC AUC
* curvas
* confusion matrix

---

## 13. Resultados iniciales

El MLP sin ajustes:

* No supera al baseline en AUC
* Tiene menor recall para la clase positiva

Esto es **esperable** en datasets tabulares pequeños.

---

## 14. Ajuste de threshold

Se analiza el efecto de distintos umbrales:

* 0.5 (default)
* 0.45
* 0.4
* 0.35
* 0.3

Conclusión:

* El threshold controla el trade-off precision vs recall
* El modelo puede ser útil aunque el AUC no cambie

---

## 15. Manejo de desbalance: `pos_weight`

Se introduce `pos_weight` en la loss:

```python
nn.BCEWithLogitsLoss(pos_weight=neg/pos)
```

Esto penaliza más los falsos negativos de la clase positiva.

Efectos observados:

* Recall de sobrevivientes ↑
* Precision ↓
* AUC ligeramente ↑ respecto al MLP original

---

## 16. Selección final de threshold

Con `pos_weight`, el mejor trade-off se obtiene alrededor de:

* **threshold ≈ 0.45**

Este punto:

* supera el recall del baseline
* mantiene métricas globales razonables

---

## 17. Conclusiones finales

* Deep learning **no es automáticamente mejor** en tabular
* El baseline es obligatorio
* El verdadero valor está en:

  * control del error
  * ajuste de costos
  * decisión consciente del threshold

Este notebook demuestra un **pipeline completo, honesto y reproducible** de clasificación binaria en PyTorch.

---

## 18. Aprendizajes clave

* Diferencia entre ranking (AUC) y decisión (threshold)
* Uso correcto de `pos_weight`
* Entrenamiento explícito en PyTorch
* Comparación justa contra baseline

Este enfoque es el esperado a nivel **senior / producción**.
