import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report

print("All libraries imported successfully!")
print("Scikit-learn version:", sklearn.__version__)
#Load
path = r"d:\old\Downloads\project\updated_pollution_dataset.xls"
df = pd.read_excel(path)
print("First 5 rows:")
print(df.head())
print("\nDataset Shape:")
print(df.shape)
print("\nColumn Names:")
print(df.columns.tolist())
print("\nDataset Information:")
df.info()

#Missing
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDuplicate Rows:")
print(df.duplicated().sum())
df = df.drop_duplicates()
print("\nDataset Shape After Removing Duplicates:")
print(df.shape)
print("\nAir Quality Categories:")
print(df["Air Quality"].value_counts())

#GRAPHS
sns.set_theme(style="whitegrid")
#Distribution of Air Quality
plt.figure(figsize=(8, 5))
sns.countplot(
    data=df,
    x="Air Quality",
    hue="Air Quality",
    order=["Good", "Moderate", "Poor", "Hazardous"],
    palette={
        "Good": "#A8D5BA",
        "Moderate": "#B8A1D9",
        "Poor": "#F2A7BB",
        "Hazardous": "#E89A8A"
    },
    legend=False
)
plt.title("Distribution of Air Quality")
plt.xlabel("Air Quality")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.show()
#Distribution of PM2.5 Levels
plt.figure(figsize=(8, 5))
sns.histplot(
    data=df,
    x="PM2.5",
    kde=True,
    color="#C3A6E8",
    edgecolor="white"
)
plt.title("Distribution of PM2.5 Levels")
plt.xlabel("PM2.5 Level")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
#Correlation Heatmap
plt.figure(figsize=(10, 7))
correlation = df.select_dtypes(include="number").corr()
sns.heatmap(
    correlation,
    annot=True,
    cmap="PuRd",
    fmt=".2f",
    linewidths=0.5
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

#Data Preprocessing
print("\nPreparing Data for Machine Learning")
X = df.drop("Air Quality", axis=1)
y = df["Air Quality"]
encoder = LabelEncoder()
y = encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data preparation completed!")
print("Training records:", X_train.shape[0])
print("Testing records:", X_test.shape[0])
print("Features:", X.shape[1])
print("Air Quality Classes:", encoder.classes_)

#Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_predictions = lr_model.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_test, lr_predictions)
print("Logistic Regression Accuracy:", round(lr_accuracy * 100, 2), "%")
print(classification_report(
    y_test,
    lr_predictions,
    labels=[0, 1, 2, 3],
    target_names=encoder.classes_,
    zero_division=0
))

#Decision Tree Classification
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)
print("Decision Tree Accuracy:", round(dt_accuracy * 100, 2), "%")
print(classification_report(
    y_test,
    dt_predictions,
    labels=[0, 1, 2, 3],
    target_names=encoder.classes_,
    zero_division=0
))

#Random Forest Classifier
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("Random Forest Accuracy:", round(rf_accuracy * 100, 2), "%")
print(classification_report(
    y_test,
    rf_predictions,
    labels=[0, 1, 2, 3],
    target_names=encoder.classes_,
    zero_division=0
))

#Support Vector Machine (SVM)
svm_model = SVC(kernel="rbf", random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_predictions = svm_model.predict(X_test_scaled)
svm_accuracy = accuracy_score(y_test, svm_predictions)
print("SVM Accuracy:", round(svm_accuracy * 100, 2), "%")
print(classification_report(
    y_test,
    svm_predictions,
    labels=[0, 1, 2, 3],
    target_names=encoder.classes_,
    zero_division=0
))

#K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
knn_predictions = knn_model.predict(X_test_scaled)
knn_accuracy = accuracy_score(y_test, knn_predictions)
print("KNN Accuracy:", round(knn_accuracy * 100, 2), "%")
print(classification_report(
    y_test,
    knn_predictions,
    labels=[0, 1, 2, 3],
    target_names=encoder.classes_,
    zero_division=0
))

#Compare all 5 algorithms
results = pd.DataFrame({
    "Algorithm": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "SVM",
        "KNN"
    ],
    "Accuracy (%)": [
        lr_accuracy * 100,
        dt_accuracy * 100,
        rf_accuracy * 100,
        svm_accuracy * 100,
        knn_accuracy * 100
    ]
})

results["Accuracy (%)"] = results["Accuracy (%)"].round(2)
print("\nAlgorithm Comparison:")
print(results.to_string(index=False))
print("\nBest Performing Algorithm:")
print(results.loc[results["Accuracy (%)"].idxmax()].to_string())

#Comparison Graph
plt.figure(figsize=(10, 6))
colors = ["#B8A1D9", "#F2A7BB", "#A8D5BA", "#E89A8A", "#A7C7E7"]
bars = plt.bar(
    results["Algorithm"],
    results["Accuracy (%)"],
    color=colors,
    edgecolor="white"
)
plt.title("Comparison of Machine Learning Algorithms", fontsize=14)
plt.xlabel("Machine Learning Algorithms")
plt.ylabel("Accuracy (%)")
plt.ylim(85, 100)
plt.xticks(rotation=15)
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.2,
        f"{height:.1f}%",
        ha="center"
    )
plt.tight_layout()
plt.show()

#Confusion Matrix — Random Forest
cm = confusion_matrix(y_test, rf_predictions)
plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Purples",
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_
)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Air Quality")
plt.ylabel("Actual Air Quality")
plt.tight_layout()
plt.show()