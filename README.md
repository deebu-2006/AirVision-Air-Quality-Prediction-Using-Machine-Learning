# 🌍 AirVision: Air Quality Prediction Using Machine Learning

**AirVision** is a machine learning project that analyzes environmental and air pollutant data to classify air quality into four categories: **Good, Moderate, Poor, and Hazardous**.

The project uses Exploratory Data Analysis (EDA), data visualization, and multiple machine learning algorithms to understand air quality patterns and evaluate classification performance.

## 🎯 Project Objectives

- Analyze environmental and air pollution data.
- Perform Exploratory Data Analysis (EDA).
- Visualize pollutant distributions and feature correlations.
- Train and compare multiple machine learning models.
- Evaluate model performance using accuracy scores and a confusion matrix.

## 📊 Dataset Overview

The dataset contains **5,000 records and 10 columns**, including environmental measurements, pollutant concentrations, and air quality labels.

| Feature | Description |
|---|---|
| Temperature | Ambient temperature |
| Humidity | Moisture level in the air |
| PM2.5 | Fine particulate matter concentration |
| PM10 | Coarse particulate matter concentration |
| NO2 | Nitrogen dioxide concentration |
| SO2 | Sulfur dioxide concentration |
| CO | Carbon monoxide concentration |
| Proximity_to_Industrial_Areas | Proximity to industrial areas |
| Population_Density | Population density |
| Air Quality | Target classification label |

### Air Quality Categories

- 🟢 Good
- 🟡 Moderate
- 🟠 Poor
- 🔴 Hazardous

## 🛠️ Technologies Used

- **Python** — Programming language
- **Pandas & NumPy** — Data handling and numerical operations
- **Matplotlib & Seaborn** — Data visualization
- **Scikit-learn** — Machine learning and model evaluation

## 🤖 Machine Learning Algorithms

Five classification algorithms were implemented and compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)

## 📈 Model Performance

| Algorithm | Accuracy |
|---|---:|
| Logistic Regression | 92.9% |
| Decision Tree | 92.8% |
| Random Forest | **95.1%** |
| Support Vector Machine (SVM) | 93.9% |
| K-Nearest Neighbors (KNN) | 92.8% |

**Best-performing model:** Random Forest, with a reported accuracy of **95.1%** on the evaluation data.

## 📉 Exploratory Data Analysis

The project includes visualizations to explore the dataset and model performance:

- Air quality category distribution
- PM2.5 concentration distribution
- Correlation heatmap of numerical features
- Model accuracy comparison
- Random Forest confusion matrix

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/AirVision.git
cd AirVision
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Run the Project

Run the Python script:

```bash
python air_quality_prediction.py
```

Make sure the dataset path in the script matches the location of your CSV file.

## 🔮 Future Scope

- Integrate real-time air quality data from monitoring APIs.
- Develop an interactive dashboard for air quality analysis.
- Evaluate the model on independent datasets from different locations.
- Explore additional features and model optimization techniques.

## 🎓 Learning Outcomes

Through this project, I gained practical experience in data preprocessing, exploratory data analysis, data visualization, machine learning classification, and model evaluation.

## 👩‍💻 Author

**B.Tech CSE — Data Science**

---

⭐ If you find this project interesting, feel free to explore the repository!
