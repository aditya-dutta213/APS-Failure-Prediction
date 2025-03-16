### **APS Failure Prediction with Imbalanced Data Techniques**  
**Author:** Aditya Dutta  
**Date:** Oct 2024 - Nov 2024  
**Technologies:** Python, SQL, Random Forest, XGBoost, SMOTE  

---

## **Project Overview**  
This project focuses on **predicting Air Pressure System (APS) failures** in Scania trucks using machine learning techniques, addressing **severe class imbalance** in the dataset. The key methods include **SMOTE for oversampling**, **Random Forests**, and **XGBoost** for classification, along with **hyperparameter tuning and feature selection**.

---

## **Dataset**  
The dataset originates from the **APS Failure at Scania Trucks** dataset available at:  
🔗 [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/APS+Failure+at+Scania+Trucks)  

- **60,000 training samples** (1,000 positive, 59,000 negative)  
- **16,000 test samples** (375 positive, 15,625 negative)  
- **171 numerical features**, including missing values  
- **Objective:** Classify APS failures (1) vs. non-failures (0)  

---

## **Methods & Techniques**  

### **1. Data Preprocessing & Feature Engineering**  
✔ **Missing Data Handling** – Mean imputation for missing values  
✔ **Feature Selection** – Coefficient of Variation (CV) used to select top features  
✔ **Correlation Analysis** – Heatmaps and scatterplots for feature importance  

### **2. Model Training**  
📌 **Random Forest Classifier**  
   - Initially trained **without compensation** for class imbalance  
   - **Out-of-Bag (OOB) error estimation**  
   - **Confusion matrix, ROC, and AUC analysis**  

📌 **XGBoost Classifier**  
   - **Hyperparameter tuning** using **GridSearchCV**  
   - Compared **uncompensated** and **SMOTE-enhanced** versions  

### **3. Handling Class Imbalance**  
🔹 **Without Compensation:** Higher accuracy but poor recall for minority class  
🔹 **With SMOTE:** Improved recall for APS failures (from **76% to 90%**), but slightly higher misclassification for non-failures  

---

## **Results & Findings**  

| Model                 | Accuracy | Recall (APS Failures) | ROC AUC |  
|----------------------|------------|---------------------|---------|  
| Random Forest (No Balancing)  | **99.2%**  | **56.6%** | **0.98** |  
| Random Forest (Balanced)  | **96.0%**  | **96.8%** | **0.99** |  
| XGBoost (No Balancing)  | **98.8%**  | **76.3%** | **0.99** |  
| **XGBoost + SMOTE** | **96.1%**  | **90.9%** | **0.99** ✅ |  

✔ **SMOTE significantly improved recall for APS failures from 76% to 90%**  
✔ **XGBoost with SMOTE provided the best balance between recall and accuracy**  

---

## **How to Run the Project**  
1️⃣ Install dependencies:  
```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn
```
2️⃣ Download the dataset from [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/APS+Failure+at+Scania+Trucks)  
3️⃣ Run the **Jupyter Notebook** or execute the script:  
```bash
python aps_failure_prediction.py
```

---

## **Key Takeaways**  
🔹 **Handling class imbalance is critical** – SMOTE significantly boosts recall  
🔹 **XGBoost outperforms Random Forest** with better handling of rare failures  
🔹 **Hyperparameter tuning and feature selection improve model robustness**  

---

## **Future Improvements**  
🚀 **Experiment with Deep Learning (LSTMs or CNNs)** for time-series APS failure data  
🚀 **Ensemble techniques** (combining RF & XGBoost)  
🚀 **Threshold tuning** to balance precision-recall trade-offs  

---

### 📌 **Contact**  
For any questions or improvements, feel free to connect! 🚀  

---

This **README.md** provides clear documentation, making it **easy for anyone to understand and reproduce your work**. Let me know if you need any modifications! 😊
