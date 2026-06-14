import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

iris = load_iris()
X = iris.data
y = iris.target

print("\nDataset Loaded")
print("Shape:", X.shape)
print("Classes:", iris.target_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel="linear", probability=True)
}

results = {}

for name, model in models.items():
    print("\n==============================")
    print("Model:", name)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print("Accuracy:", acc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=iris.target_names)
    disp.plot(cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.show()

    # Save model (INDUSTRY FEATURE)
    with open(f"{name}_model.pkl", "wb") as f:
        pickle.dump(model, f)

plt.figure(figsize=(6,4))
plt.bar(results.keys(), results.values(), color=["#1e3a8a", "#2563eb"])
plt.title("Model Performance Comparison")
plt.ylim(0, 1)
plt.ylabel("Accuracy")

for i, v in enumerate(results.values()):
    plt.text(i, v + 0.02, f"{v:.2f}", ha="center")

plt.show()

plt.figure(figsize=(7,5))

plt.scatter(
    X[:, 0],
    X[:, 2],
    c=y,
    cmap="viridis",
    s=70,
    edgecolor="k"
)

plt.title("Feature Distribution (Industry-Level Visualization)")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.colorbar(label="Class")
plt.grid(alpha=0.3)
plt.show()

print("\nModels saved as .pkl files (industry practice)")