# DA Assignment 2
### Aravindhan Mohanraj | Roll No: DA25S006

## 📌 Overview  
This assignment explores **dimensionality reduction, visualization, and classification** using the **Mushroom Dataset**. The dataset classifies mushrooms as **edible** or **poisonous** based on categorical attributes.  
The main objective is to apply **Principal Component Analysis (PCA)** to handle feature redundancy and evaluate its impact on classification performance compared to the original high-dimensional data.

---

## 🔑 Steps Performed  
1. **Data Preprocessing & EDA**  
   - Handled missing values (encoded as `?`).  
   - Replaced missing markers with `"missing"`.  
   - Applied **One-Hot Encoding** to convert categorical features into numeric format.  
   - Explored class balance between edible and poisonous mushrooms.  

2. **Dimensionality Reduction (PCA)**  
   - Standardized features for consistency.  
   - Applied **PCA** to reduce dimensionality while retaining variance.  
   - Visualized principal components to identify separability between classes.  

3. **Classification (Logistic Regression)**  
   - Trained Logistic Regression models on both:  
     - Original dataset.  
     - PCA-transformed dataset.  
   - Compared performance metrics (accuracy, confusion matrix, classification report).  

4. **Performance Evaluation**  
   - Assessed whether PCA improved classification by reducing redundancy.  
   - Discussed trade-offs between interpretability and efficiency.  

---

## 📊 Tools & Libraries  
- **Python**, **Pandas**, **NumPy**  
- **Matplotlib**, **Seaborn** (visualization)  
- **Scikit-learn** (PCA, Logistic Regression, metrics)  

---

## 🚀 Results & Insights  
- PCA successfully reduced dataset dimensionality and provided better visualization.  
- Logistic Regression performed comparably on both datasets, showing PCA can simplify computation without major performance loss.  
- Demonstrated that PCA helps in handling **feature collinearity and redundancy** effectively.  

---

## 📂 Repository Structure  
```
├── DA_Assignment_2.ipynb   # Jupyter notebook with full code & outputs
├── README.md               # Project documentation (this file)
└── data/                   # Mushroom dataset (not included, load externally)
```

---

## 📝 Author  
**Aravindhan Mohanraj**  
Roll No: **DA25S006**  
DA Assignment 2
