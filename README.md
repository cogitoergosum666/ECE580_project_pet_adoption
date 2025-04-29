# Data-Driven Discoveries in Pet Adoption Patterns

[GitHub Repository](https://github.com/cogitoergosum666/ECE580_project_pet_adoption)

## 📚 Project Overview

This project investigates how structured pet profile features influence adoption speed, leveraging the **PetFinder.my Adoption Prediction** dataset.  
We systematically benchmark multiple machine learning models, aiming to balance predictive performance with model interpretability.  
Our work is conducted as a final project for **ECE580** at Duke University.

## 🔍 Main Objectives
- Predict pet adoption speed using structured attributes (e.g., age, breed, sterilization status, fee).
- Evaluate if flexible models (Random Forest, XGBoost) significantly outperform simpler interpretable models (LogisticGAM, Decision Trees).
- Analyze feature importance across different modeling frameworks to understand key adoption drivers.

## 🛠 Models Evaluated
- **Logistic Generalized Additive Model (LogisticGAM)**: for binary classification (fast vs slow adoption).
- **Linear Generalized Additive Model (LinearGAM)**: for ordinal classification (0 to 4 adoption speeds).
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **XGBoost Classifier**

## ⚙️ Methodology
- **Data Preprocessing**:  
  - Target encoding for high-cardinality categorical features (breed, color, state).
  - One-hot encoding for low-cardinality categorical features (pet type, gender).
  - Ordinal encoding for ordered features (maturity size, health condition).
  - Standard scaling for numerical features (age, adoption fee, media counts).

- **Model Training and Hyperparameter Tuning**:  
  - LogisticGAM and LinearGAM: regularization via grid search.
  - Decision Tree: maximum depth tuning.
  - Random Forest: randomized search over tree count, depth, feature sampling.
  - XGBoost: cross-validation to tune learning rate, gamma, subsampling ratios.

- **Evaluation Metrics**:  
  - Validation Accuracy  
  - Weighted F1 Score  
  - Confusion Matrices  
  - Drop-in-Accuracy (for GAMs)  
  - Feature Importance Analysis

## 📈 Key Results

| Model           | Accuracy | Weighted F1 Score |
|-----------------|----------|-------------------|
| LogisticGAM     | 66.4%    | 66.2%             |
| LinearGAM       | 31.1%    | 31.2%             |
| Decision Tree   | 35.0%    | 35.0%             |
| Random Forest   | 42.0%    | 41.0%             |
| XGBoost         | 41.4%    | 39.8%             |

- **Random Forest** achieved the highest validation accuracy among ensemble methods.
- **LogisticGAM** offered the best trade-off between interpretability and performance in binary classification.
- **Feature Importance** highlighted that **Age**, **Fee**, **Photo Amount**, and **Breed** are critical adoption drivers.

## 🧠 Key Insights
- Structured features (without rich textual or image data) can moderately predict adoption outcomes (~40-44% accuracy).
- Biological attributes (age, breed) and marketing factors (adoption fee, number of media) strongly affect adoption likelihood.
- Ensemble models (Random Forest, XGBoost) handle complex feature interactions better than simpler models.

## 📌 Limitations
- Predictive performance on minority classes (especially class 0) remains weak across all models.
- Structured data alone cannot fully capture adopter preferences or pet personality.

## 🚀 Future Work
- Incorporate richer textual features (pet descriptions) using NLP techniques.
- Experiment with **CatBoost** or **LightGBM** for potential accuracy gains.
- Explore multi-task learning frameworks to jointly predict multiple adoption outcomes.

## 👥 Authors
- **Yuqi Su** (ys457)  
- **Haiwei Du** (hd165)

Duke University, ECE580, Spring 2024.
