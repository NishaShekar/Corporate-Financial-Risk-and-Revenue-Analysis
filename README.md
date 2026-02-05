# Corporate-Financial-Risk-and-Revenue-Analysis

An end-to-end **machine learning–based corporate revenue forecasting application** built as part of a **PGP in Data Science (Capstone Project)**. This project combines rigorous financial data modeling with an interactive **Streamlit web application** to deliver accurate, interpretable, and scalable revenue predictions for corporate decision-making.

---

## 🚀 Project Overview

Accurate revenue forecasting is critical for budgeting, investment planning, and financial risk assessment. Traditional statistical methods often fail to capture the **non-linear relationships**, **skewed distributions**, and **outliers** present in real-world corporate finance data.

This project addresses these challenges by:

* Applying robust **data preprocessing and feature engineering**
* Comparing multiple regression models
* Selecting **XGBoost** as the final high-performing model
* Deploying the trained model through an **interactive Streamlit dashboard**

---

## 🎯 Key Objectives

* Predict **annual corporate revenue** using financial indicators
* Prevent **target leakage** and ensure realistic forecasting
* Identify **key financial drivers** influencing revenue
* Provide a **user-friendly tool** for both analysts and non-technical stakeholders

---

## 🧠 Modeling Approach

### Models Evaluated

* **Linear Regression** (Baseline)
* **Random Forest Regressor**
* **XGBoost Regressor** ✅ *(Final Model)*

### Why XGBoost?

* Highest performance: **R² = 0.994**
* Lowest prediction error (RMSE & MAE)
* Excellent handling of non-linear financial relationships
* Robust to skewness and outliers

### Key Techniques Used

* Median imputation for missing values
* IQR-based outlier capping
* Lag feature engineering (Revenue, Profit, Cash Flow)
* Time-based train–test split to avoid leakage

---

## 📊 Features of the Application

### 🔹 Single Prediction Mode

* Input company financial metrics manually
* Get instant revenue prediction
* View:

  * Revenue gauge chart
  * Feature importance explanation

### 🔹 Batch Prediction Mode

* Upload a CSV file with multiple company records
* Generate bulk revenue forecasts
* Download results as a CSV file

### 🔹 Explainability

* Feature importance visualization to show **why** a prediction was made

---

## 🗂️ Project Structure

```
├── app.py                     # Streamlit web application
├── FinalModel.ipynb           # Model development & evaluation notebook
├── revenue_model.pkl          # Trained XGBoost model
├── model_features.pkl         # Feature list used by the model
├── requirements.txt           # Python dependencies
├── PGPDSE FT Capstone Project final.pdf  # Detailed project report
└── README.md                  # Project documentation
```

---

## ⚙️ Tech Stack

* **Python**
* **Streamlit** – Web application
* **Pandas, NumPy** – Data processing
* **Scikit-learn** – ML utilities
* **XGBoost** – Final prediction model
* **Plotly** – Interactive visualizations
* **Joblib** – Model serialization

---

## 🧪 How to Run the Project Locally

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/finvision-revenue-forecasting.git
cd finvision-revenue-forecasting
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

---

## 📈 Business Impact

* Enables **data-driven budgeting & investment planning**
* Identifies early **financial risk signals**
* Demonstrates the superiority of **ensemble ML methods** in finance
* Bridges the gap between **data science models and real-world usability**

---

## ⚠️ Limitations

* Dataset limited to financial years **2014–2016**
* No macroeconomic indicators included
* Potential overfitting due to high model accuracy

---

## 🔮 Future Enhancements

* Extend dataset to longer time horizons
* Integrate macroeconomic indicators (GDP, inflation)
* Experiment with time-series models (Prophet, LSTM)
* Deploy as a cloud-based real-time application

---



## 📜 License

This project is for **academic and educational purposes**.

---

⭐ *If you find this project useful, feel free to star the repository!*
