# Project 2: Data Classification & Industry-Level Visualization Pipeline

An end-to-end Machine Learning classification architecture targeting the structural Iris dataset. The pipeline integrates data normalization, multi-model evaluation frameworks (KNN vs. SVM), automated serialization patterns, and production-ready metric graphs.

---

## 🛠️ Tech Stack & Advanced Features
- **Core Frameworks:** Scikit-Learn, NumPy, Pandas.
- **Data Preprocessing:** Implements `StandardScaler` to handle distribution feature scaling and prevent distance-metric bias.
- **Algorithms Implemented:**
  - **K-Nearest Neighbors (KNN):** Configured with $k=5$ for neighborhood boundary determination.
  - **Support Vector Classifier (SVC):** Built using a linear kernel constraint and probabilistic execution flags.
- **Model Serialization:** Auto-saves operational weights into binary storage artifacts (`.pkl`) for native inference integrations.
- **Analytical Plots:** Includes Confusion Matrix displays, relative performance bar comparisons, and multi-feature cluster scatter configurations.

---

## 📂 Repository Matrix
- `dataclassification.py` — Complete end-to-end Python pipeline (Data loading, train-test splitting, training execution, evaluation reports, and graph generation layers).
- `KNN_model.pkl` — Serialized Production-ready K-Nearest Neighbors model artifact.
- `SVM_model.pkl` — Serialized Production-ready Support Vector Machine model artifact.

---

## 🚀 Local Execution & Benchmarking

To launch the training cycle and view dataset visualizations, deploy the following execution trigger in your terminal:

```bash
# Initialize data pipelines and fit training spaces
python "dataclassification.py"
