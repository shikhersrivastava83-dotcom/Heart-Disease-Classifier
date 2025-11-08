# Heart Disease Classifier 💓

A machine learning project that predicts the likelihood of heart disease based on patient health data.  
This project compares the performance of **Decision Tree** and **Random Forest** algorithms on a real-world dataset.

---

## 🧠 Overview
This project classifies whether a person is likely to have heart disease based on medical attributes such as age, cholesterol level, resting blood pressure, chest pain type, and more.  

Two supervised learning models were trained and evaluated:
- **Decision Tree Classifier**
- **Random Forest Classifier**

---

## 📊 Dataset
- **Source:** [Kaggle - Heart Disease Dataset](https://www.kaggle.com/)  
- **Samples:** 2833 records  
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

## 📦 How to Run
Open your terminal (or command prompt) and run the following commands one by one:

```bash
# Clone the repository
git clone https://github.com/sikhersrivastava83-dotcom/heart-disease-classifier.git

# Move into the project folder
cd heart-disease-classifier

# Install dependencies
pip install pandas NumPy scikit-learn

# Run the classifier
python CARDIO_DISEASE.py



