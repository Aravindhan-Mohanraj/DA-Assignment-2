# DA Assignment 2  
**Name:** Aravindhan Mohanraj  
**Roll No:** DA25S006  

---

## 📌 Overview  
This assignment applies **Principal Component Analysis (PCA)** on the **Mushroom Dataset** to study dimensionality reduction and its effect on classification performance. Logistic Regression is used as the classifier to compare results on the **original vs PCA-transformed data**.

---

## 🔑 Workflow  
1. **Data Preprocessing**  
   - Handled missing values (`?` → `"missing"`).  
   - One-hot encoded categorical features.  

2. **Dimensionality Reduction (PCA)**  
   - Standardized features.  
   - Reduced dimensions while retaining most variance.  
   - Visualized separation using principal components.  

3. **Classification & Evaluation**  
   - Trained Logistic Regression on both datasets.  
   - Compared accuracy, confusion matrix, and classification reports.  
   - Analyzed PCA’s impact on performance and redundancy handling.  

---

## 📊 Tools Used  
- Python (Pandas, NumPy, Scikit-learn)  
- Matplotlib, Seaborn  

---

## 🚀 Key Insights  
- PCA reduced dimensionality effectively and improved visualization.  
- Logistic Regression gave similar accuracy on both datasets, showing PCA simplifies computation without major performance loss.  
- PCA is useful in handling **feature collinearity and redundancy**.  
