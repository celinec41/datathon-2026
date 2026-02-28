# 📊 Predicting Financial Impact of Economic Shocks Using Machine Learning

## 🔍 Overview

This project builds a supervised machine learning model to predict how an individual's financial situation would change during a major economic shock similar to COVID-19.

Given demographic and financial characteristics, the model predicts one of three outcomes:

* **1 — Improved**
* **2 — Worsened**
* **3 — Stayed Same**

The final output functions as a simple prediction tool:
a user enters their financial profile, and the model returns the most likely shock outcome.

---

## 🎯 Research Question

> Which financial and demographic characteristics predict whether an individual’s financial situation improves, worsens, or remains stable during an economic shock?

We further examine:

* How predicted outcomes vary across provinces
* How vulnerability differs across age groups
* Which financial factors most strongly drive financial deterioration

---

## 📂 Dataset

Source: **Survey of Financial Security (SFS)**

**Target Variable**

* `PATTSITC`

  * 1 = Improved
  * 2 = Worsened
  * 3 = Stayed Same

**Key Input Features**

* Age group
* Province
* Education level
* After-tax income
* Homeownership status
* Mortgage debt
* Student loan debt
* Credit card debt
* Line of credit
* Bank deposits
* TFSA balance

---

## 🧠 Methodology

### 1️⃣ Data Preprocessing

* Cleaned missing values
* Encoded categorical variables using One-Hot Encoding
* Split dataset into 80% training / 20% testing

### 2️⃣ Model Development

We trained multi-class classification models:

* **Multinomial Logistic Regression**
* **Random Forest**
* (Optional) XGBoost

### 3️⃣ Evaluation Metrics

* Accuracy
* Macro F1-Score
* Confusion Matrix

---

## 🤖 Prediction Logic

The trained model estimates probabilities:

[
P(Improved), \quad P(Worsened), \quad P(Stayed\ Same)
]

The predicted class is:

[
\text{Prediction} = \arg\max(P)
]

The tool returns:

* Predicted category (1, 2, or 3)
* Associated probability (confidence score)

---

## 🧪 Example Prediction

**User Input**

* Age Group: 26–35
* Province: Ontario
* Income: $55,000
* Credit Card Debt: $8,000
* Student Loan: $15,000
* Savings: $2,500
* Homeowner: No

**Model Output**

Predicted Outcome:
**2 — Worsened**
Confidence: 0.64

---

## 📈 Key Findings

* High unsecured debt (credit card & line of credit) strongly predicts worsening outcomes.
* Low liquidity significantly increases vulnerability.
* Province remains a statistically significant factor after controlling for income and debt.
* Younger age groups exhibit higher predicted vulnerability during economic shocks.

---

## 📁 Project Structure

```
├── final_submission.ipynb
├── data/
│   └── personal_finance_dataset.xlsx
├── README.md
```

---

## ▶️ How to Run

1. Install required packages:

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn
```

2. Open the notebook:

```bash
jupyter notebook final_submission.ipynb
```

3. Run all cells to train the model and test custom user inputs.

---

## ⚠️ Limitations

* Outcome variable is self-reported financial perception.
* The model does not include macroeconomic indicators.
* Predictions are for analytical purposes only and should not be interpreted as financial advice.

---

## 🛡 Ethical Considerations

* No personally identifiable information was used.
* Model bias across provinces and age groups was evaluated.
* The system is not intended for real credit or lending decisions.

---

# 🏆 Why This Project Matters

By identifying which households are most vulnerable to economic shocks, this model provides actionable insights that can inform targeted financial support programs, policy design, and resilience planning across Canadian provinces and life stages.

