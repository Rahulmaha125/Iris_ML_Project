# 🌸 Iris Dataset Classification – End-to-End Machine Learning Project

An end-to-end Machine Learning project that classifies Iris flower species based on sepal and petal measurements using multiple classification algorithms and deploys an interactive web interface with **Streamlit**.

---

## 📌 Project Overview

The **Iris Dataset** is a classic dataset in machine learning consisting of 150 samples of iris flowers categorized into 3 species:
1. **Setosa** (`Class 0`)
2. **Versicolor** (`Class 1`)
3. **Virginica** (`Class 2`)

### 📐 Features Included:
- **Sepal Length** (cm)
- **Sepal Width** (cm)
- **Petal Length** (cm)
- **Petal Width** (cm)

---

## 🚀 Key Features & Highlights

- **Data Preprocessing & Cleaning**: Missing value check, duplicate removal, feature scaling using `StandardScaler`.
- **Exploratory Data Analysis (EDA)**: Visualizations including Pairplots, Boxplots, Histograms, Violin plots, Scatter plots, and Correlation Heatmaps.
- **Multi-Model Training**:
  - Logistic Regression
  - Decision Tree Classifier
  - k-Nearest Neighbors (k-NN)
- **Model Evaluation**: Accuracy Score, Confusion Matrix, and Classification Reports.
- **Model Serialization**: Saved trained model (`iris_model.pkl`) and scaler (`iris_scaler.pkl`) using `pickle`.
- **Interactive Web App**: Built with **Streamlit** for real-time predictions.

---

## 📁 Project Structure

```
Iris_ML_Project/
│
├── Iris.csv                 # Raw dataset file
├── Iris_ML_Project.ipynb    # End-to-end Jupyter Notebook pipeline
├── app.py                   # Streamlit Web Application
├── iris_model.pkl           # Trained Logistic Regression pickle model
├── iris_scaler.pkl          # Standardized feature scaler object
└── README.md                # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone or Open Project Directory
```bash
cd D:/Iris_ML_Project
```

### 2. Install Required Dependencies
Ensure Python 3.8+ is installed, then run:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit
```

---

## 💻 How to Run

### Option 1: Run the Jupyter Notebook
Open [`Iris_ML_Project.ipynb`](Iris_ML_Project.ipynb) in VS Code or Jupyter Notebook to view data exploration, model training, and evaluation steps.

### Option 2: Run the Streamlit Web Application
Launch the interactive web application by running:

```bash
python -m streamlit run app.py
```

The web app will open automatically in your browser at `http://localhost:8501`.

---

## 📊 Model Evaluation Results

| Model | Accuracy | Status |
| :--- | :---: | :---: |
| **Logistic Regression** | **100% (1.00)** | Selected for Deployment |
| **Decision Tree** | **100% (1.00)** | Evaluated |
| **k-Nearest Neighbors (k-NN)** | **100% (1.00)** | Evaluated |

---

## 🤝 Acknowledgments
- **Dataset**: `sklearn.datasets.load_iris` / UCI Machine Learning Repository
- **Organization**: IICET Services Private Limited (Classification Project No:01)
