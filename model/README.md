---
language: en
license: mit
library_name: scikit-learn
tags:
  - python
  - package-compatibility
  - prediction
  - scikit-learn
  - tabular-classification
metrics:
  - accuracy
  - f1
model-index:
  - name: pycompat-predictor
    results:
      - task:
          type: tabular-classification
          name: Package Compatibility Prediction
        metrics:
          - name: Accuracy
            type: accuracy
            value: 0.9708
          - name: F1 Score
            type: f1
            value: 0.97
---

# PyCompat — Python Package Compatibility Predictor

AI model that predicts whether a Python package version is compatible with a given system
(OS, Python version, platform) and recommends the best compatible versions.

## Model Details

- **Model Type:** Random Forest (compatibility) + Gradient Boosting (error type)
- **Training Data:** 5484 compatibility test records
- **Packages:** 198 unique packages
- **Python Versions:** 3.10, 3.11, 3.12, 3.9
- **Platforms:** darwin_x86_64

## Performance

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Compatibility | 0.9708 | 0.97 |
| Error Type | 0.9836 | 0.9826 |

## Usage

```python
from pycompat_model import PyCompatModel

# Load model
model = PyCompatModel.load("./model")

# Single prediction
result = model.predict("boto3", "1.42.49", "3.12", "darwin_x86_64")
print(result)
# {'is_compatible': True, 'confidence': 0.9977, 'predicted_error_type': 'none', ...}

# Get recommendations
recs = model.recommend("alembic", "3.9")
for r in recs:
    status = "✅" if r["is_compatible"] else "❌"
    print(f"  v{r['version']} {status} ({r['confidence']:.0%})")

# Batch prediction
results = model.predict_batch([
    {"package": "boto3", "version": "1.42.49", "python_version": "3.12"},
    {"package": "alembic", "version": "1.18.4", "python_version": "3.9"},
])
```

## Error Types Predicted

| Error Type | Description |
|-----------|-------------|
| `none` | Fully compatible |
| `no_wheel` | No compatible wheel/distribution found |
| `import_error` | Installs but fails to import |
| `abi_mismatch` | ABI incompatibility with dependencies |
| `build_error` | Failed to build from source |
| `timeout` | Network timeout during install |

## Training

```python
from pycompat_model import PyCompatModel

model = PyCompatModel.train_from_data("data.json")
model.save("./model")
```
