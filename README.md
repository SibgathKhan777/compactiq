# 🧬 CompactIQ (PyCompat)

**An AI-Powered Python Package Compatibility Prediction Engine**

CompactIQ (also referred to as PyCompat) is a complete machine learning system designed to predict whether a specific Python package version will successfully install and import on a given system architecture and Python version.

If a package is incompatible, the model predicts the exact **Error Type** (e.g., `no_wheel`, `abi_mismatch`, `import_error`). It can also rank and recommend the best alternative versions of a package for your system.

---

## 🚀 Features
- **Binary Compatibility Model (Random Forest)**: 97.1% accuracy.
- **Error Type Classifier (Gradient Boosting)**: 98.4% accuracy.
- **Flask REST API**: Production-ready API server to integrate predictions anywhere.
- **Web Dashboard**: Premium dark-themed UI with interactive analytics and a compatibility checker.
- **Hugging Face Hub**: Supports one-click download/upload to huggingface (`sibbbuu/pycompat-model`).

## 💻 Quick Start

### 1. Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```
Or install the local package directly:
```bash
pip install -e .
```

### 2. Start the API Server
Run the production REST API to serve predictions (defaults to port 8080):
```bash
python api_server.py --port 8080
```

### 3. Start the Web Dashboard
Launch the premium interactive dashboard to visualize stats and manually test versions:
```bash
python app.py
```
*Visit `http://localhost:5050` in your browser.*

### 4. CLI Usage
You can test predictions directly from the terminal without starting a server:
```bash
# General predict format:
# python pycompat_model.py predict <model_dir> <package> <version> <python_ver>
python pycompat_model.py predict ./model boto3 1.42.49 3.12

# Get recommendations for the best versions:
python pycompat_model.py recommend ./model alembic 3.9
```

---

## 📊 Gantt Chart Generation

If you are planning out the development lifecycle of this project (or a similar AI project) and want to generate a **Gantt Chart** exactly like the classic Data Science timeline example, you can copy and paste the prompt below into ChatGPT, Claude, or Gemini. 

It is engineered to generate a Python script using Matplotlib that beautifully recreates the exact styling, colors, and layout of a professional project timeline.

### 📋 The Prompt:
```text
Write a Python script using Matplotlib to generate a horizontal Gantt Chart timeline for an AI prediction project. 

It must exactly match this visual style:
1. The background of the plot and the figure should be a light grey color (e.g., '#cfcfcf' or '#d9d9d9').
2. There should be a massive main title at the top saying "GANTT CHART" in a dark blue font (e.g., '#0b4d8c'), with a smaller subtitle below it saying "Project Timeline (Apr - Nov)".
3. Use horizontal bars (`ax.barh`). 
4. The X-axis should represent the months from April to November. The labels should perfectly align with the start of each month (Apr, May, Jun, Jul, Aug, Sep, Oct, Nov).
5. The Y-axis should list the project tasks in descending order from top to bottom.
6. The tasks and their approximate timelines should be:
   - "Research & Planning" (Apr to mid-May)
   - "Data Collection & Integration" (Apr to Jun)
   - "Data Preprocessing" (May to Jun)
   - "Feature Engineering" (May to Jul)
   - "Baseline Model" (Jun to Jul)
   - "Advanced Model Development" (Jun to Aug)
   - "Model Comparison & Evaluation" (Jul to Aug)
   - "Dashboard UI & API Integration" (Aug to Oct)
   - "System Integration" (Sep to Oct)
   - "Performance Optimization" (Sep to Nov)
   - "Testing & Validation" (Oct to Nov)
   - "Documentation & Report" (Oct to Nov)
   - "Final Demo & Closure" (Nov)
7. Give every single task a distinct, solid color (e.g., dark blue, orange, green, red, purple, brown, pink, dark grey, olive, teal).
8. Put a thin black border frame around the main plot area.
9. Ensure there are no gridlines, and keep the font sizes legible (around 9 or 10 for the Y-axis tasks).

Provide the fully working Python code.
```
*Just copy the block above and the AI will generate the Python code required to plot your chart!*

---

## 🌐 API Endpoints
When the API is running, the following REST endpoints are available:

- `POST /api/predict` — Check compatibility for a specific package version.
- `POST /api/predict/batch` — Send an array of configurations to test at once.
- `POST /api/recommend` — Pass a package and Python version to get a ranked list of versions.
- `POST /api/validate` — Validate (and auto-correct) a whole `pip install` line: ML prediction + live PyPI verification + joint dependency conflict checking across every package, with a risk score and explanation. See below.
- `GET /api/info` — Fetch model metrics and accuracy stats.
- `GET /api/packages` — Browse the list of supported pip packages.

### `POST /api/validate`

Validates every `package==version` pin in a snippet of `pip install` code — not just individually, but against each other (do the pinned versions violate each other's declared dependencies?) and against live PyPI (does the version actually exist, does a wheel exist for this platform/Python combo?). Returns a risk score, a plain-language explanation, and a corrected install line.

```bash
curl -X POST http://localhost:8080/api/validate \
  -H "Content-Type: application/json" \
  -d '{
    "code": "pip install alembic==1.18.4 SQLAlchemy==1.0.0",
    "python_version": "3.12",
    "platform": "darwin_x86_64"
  }'
```

```json
{
  "original_code": "pip install alembic==1.18.4 SQLAlchemy==1.0.0",
  "corrected_code": "pip install alembic==1.18.4 SQLAlchemy==2.0.42",
  "changed": true,
  "risk_score": 1.0,
  "joint_dependency_conflicts": [
    {
      "from_package": "alembic",
      "depends_on": "SQLAlchemy",
      "required_specifier": ">=1.4.23",
      "pinned_version": "1.0.0",
      "satisfied": false
    }
  ],
  "packages": [ { "package": "alembic", "is_clean": true, "..." : "..." },
                { "package": "SQLAlchemy", "is_clean": false, "explanation": "...", "..." : "..." } ]
}
```

Set `"live": false` in the request body to skip the live PyPI calls and get an ML-only response (useful offline).

### Standalone modules behind `/api/validate`

These can also be run directly, without the API server:

- `live_verify.py` — checks a package/version against PyPI right now (does it exist, is there a wheel for this platform + Python version). Fills the gap where the trained model can only reflect what was true in `data.json` at training time.
  ```bash
  python live_verify.py boto3 1.42.49 3.12 darwin_x86_64
  ```
- `dependency_conflicts.py` — checks a batch of pinned packages against each other's real PyPI-declared `requires_dist`, catching cross-package conflicts a single-package classifier can't see.
  ```bash
  python dependency_conflicts.py 3.12 alembic 1.18.4 SQLAlchemy 1.0.0
  ```
- `eval_temporal_holdout.py` — evaluates the model on packages/versions it was never trained on (instead of the random 80/20 split), to measure real-world generalization rather than in-distribution fit.
  ```bash
  python eval_temporal_holdout.py data.json
  ```

## 🤝 Contributing
Contributions are welcome. Please ensure that modifying the dataset (`data.json`) triggers a successful retraining. By default, `app.py` actively monitors `data.json` and automatically retrains the model in the background when changes are detected!
