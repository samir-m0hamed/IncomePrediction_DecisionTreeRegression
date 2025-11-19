# 📊 Income Prediction Using Decision Tree Regression

This project aims to predict individual income using a complete Machine Learning workflow.  
The dataset contains demographic, education, employment, and household-related attributes, and the model uses a Decision Tree Regressor with full hyperparameter tuning to estimate income values.

---

## 🚀 Project Overview
- Perform data cleaning and preprocessing.
- Apply Ordinal and One-Hot Encoding to categorical features.
- Use log transformation to reduce skewness in the target variable.
- Split data into training and testing sets.
- Use **GridSearchCV** to optimize tree hyperparameters.
- Evaluate model performance (R², RMSE).
- Visualize predicted vs actual income values.

---

## 🛠️ Technologies Used
- Python  
- Pandas  
- NumPy  
- Scikit-Learn  
- Plotly  
- Jupyter Notebook
- Google Colab

---

## 🔧 Model Details
- **Algorithm:** Decision Tree Regression  
- **Tuning:**  
  - `max_depth`  
  - `min_samples_leaf`  
  - `min_samples_split`  
- **Scoring:** Negative Mean Squared Error (MSE)  
- **Target:** Income (log-transformed during training)

---

## 📈 Results
Due to the synthetic nature of the dataset, the model shows:
- **Training R²:** low  
- **Testing R²:** low  
This indicates **underfitting**, meaning the dataset lacks strong relationships between features and income.

Despite this, the project demonstrates a clean, end-to-end ML pipeline suitable for learning and experimentation.

---

## 📁 Files Included
data.csv → Dataset used for training and tetsing the model
Income_Prediction.ipynb → Main notebook containing full ML workflow
README.md → Project documentation

---

## 📈 Results
Due to the synthetic nature of the dataset and the selected model ( Decision Tree Regression ) , the model shows **underfitting**, with low R² scores on both training and testing sets.  
This demonstrates a realistic challenge when datasets lack strong feature–target relationships.

---

## 🎯 Future Improvements
- Try ensemble models: RandomForest, GradientBoosting, XGBoost  
- Use a more realistic dataset  
- Apply advanced feature engineering to extract meaningful patterns  

---

## 👤 Author
Developed by **Samir Mohamed** as part of a regression machine learning practice project.
