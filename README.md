# Heart Disease Classifier 💓  

A machine learning project that predicts the likelihood of heart disease based on patient health data.  
This project compares the performance of **Decision Tree** and **Random Forest** algorithms on a real-world dataset.

---

## 🧠 Overview
The goal of this project is to classify whether a person is likely to have heart disease based on medical attributes such as age, cholesterol level, resting blood pressure, chest pain type, and more.  

Two supervised learning models were trained and evaluated:
- **Decision Tree Classifier**
- **Random Forest Classifier**

---

## 📊 Dataset
- **Source:** [Kaggle - Heart Disease Dataset](https://www.kaggle.com/)  
- **Samples:** 3108 records  
- **Preprocessing:**
  - Missing values handled
  - Categorical features encoded numerically
  - Features scaled for consistency
  - Chest pain type column normalized from values 1–4 → 0–3 for consistency

---

## ⚙️ Methods
- Data preprocessing and feature encoding with `pandas` and `scikit-learn`
- Model training using `DecisionTreeClassifier` and `RandomForestClassifier`
- Model validation on separate validation and test sets
- Performance measured via accuracy

---

## 📈 Results

| Model           | Train Accuracy | Validation Accuracy | Test Accuracy |
|-----------------|----------------|----------------------|---------------|
| Decision Tree   | 90.47%         | 90.00%              | 86.17%        |
| Random Forest   | 93.85%         | 90.65%              | 89.39%        |

**Random Forest** achieved the highest and most consistent accuracy across all sets.

---

## 🧩 Tools & Libraries
- Python  
- NumPy  
- Pandas  
- Scikit-learn  

---

## 🔍 Insights
- Random Forest provided better generalization with less variance between training and test accuracies.  
- Feature importance helped identify which health metrics contribute most to prediction accuracy.  
- Demonstrates how ensemble methods outperform a single decision tree on structured medical data.

---

## 📦 How to Run
Clone this repository:
```bash
git clone https://github.com/shikhersrivastava83-dotcom/Heart-disease-classifier.git

1. Clone this repository:
2. ```bash
   git clone https://github.com/shikhersrivastava83-dotcom/Heart-Disease-Classifier.git
   ```bash
   git clone https://github.com/<your-username>/heart-disease-classifier.git
