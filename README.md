# PrimeEdge Lending: Loan Default Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-green)
![Status](https://img.shields.io/badge/Status-Complete-success)

## Project Overview

**PrimeEdge Lending** is experiencing a critical business challenge with a **66.86% loan default rate**, significantly impacting financial stability. This project develops and evaluates multiple machine learning models to predict loan defaults and improve the loan approval process.

### Business Problem

- **Current Default Rate**: 66.86%
- **Manual Approval Process**: Subjective, inconsistent decision-making
- **High-Risk Approvals**:
  - Low credit scores (300-500): 85.23% default rate
  - Personal loans: 69.28% default rate
- **Rule-Based System**: Only 1.2% approval rate with 66% precision

### Solution

Build and compare **6 machine learning classification models** to:
1. Predict loan default probability
2. Reduce financial losses
3. Improve risk assessment
4. Enable data-driven, consistent decisions
5. Increase profitable lending opportunities

---

## Table of Contents

- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Business Impact](#business-impact)
- [Key Findings](#key-findings)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

---

## Dataset

**File**: `Loan_Delinquent_Analysis_Dataset.csv`

**Shape**: 11,548 rows × 10 columns

**Features**:
| Column | Description | Type |
|--------|-------------|------|
| ID | Unique identifier | Integer |
| Delinquency_Status | Target variable (1=Default, 0=No Default) | Binary |
| Loan_Term | Duration of loan (months) | Categorical |
| Borrower_Gender | Gender of borrower | Categorical |
| Age_Group | Age group of borrower | Categorical |
| Loan_Purpose | Reason for loan | Categorical |
| Home_Status | Home ownership status | Categorical |
| Credit_Score_Range | Credit score range | Categorical |
| Income | Annual income ($1000s) | Numerical |
| Loan_Amount | Loan amount ($) | Numerical |

**Data Quality**:
- No missing values
- Clean dataset
- Minor case sensitivity issue in Loan_Purpose (fixed in preprocessing)

---

## Project Structure

```
PrimeEdge_Lending/
│
├── Loan_Delinquent_Analysis_Dataset.csv      # Raw dataset
├── PrimeEdge_Lending_Loan_Default_Prediction.ipynb  # Main analysis notebook
├── README.md                                   # Project documentation
├── explore_data.py                            # Data exploration script
│
├── Models/ (generated after running notebook)
│   ├── best_loan_default_model.pkl           # Trained Random Forest model
│   ├── scaler.pkl                             # StandardScaler for features
│   └── label_encoders.pkl                     # Label encoders for categorical variables
│
└── requirements.txt                           # Python dependencies
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- pip package manager

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/PrimeEdge-Lending.git
   cd PrimeEdge-Lending
   ```

2. **Create virtual environment** (optional but recommended):
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

5. **Open the notebook**: `PrimeEdge_Lending_Loan_Default_Prediction.ipynb`

---

## Usage

### Running the Analysis

1. Open `PrimeEdge_Lending_Loan_Default_Prediction.ipynb`
2. Run all cells sequentially (Cell → Run All)
3. The notebook will:
   - Load and explore the dataset
   - Clean and preprocess data
   - Perform exploratory data analysis
   - Train 6 different ML models
   - Compare model performance
   - Save the best model

### Making Predictions with Saved Model

```python
import pickle
import pandas as pd

# Load saved artifacts
with open('best_loan_default_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Example: New applicant data
new_applicant = {
    'Loan_Term': '36 months',
    'Borrower_Gender': 'Male',
    'Loan_Purpose': 'House',
    'Home_Status': 'Mortgage',
    'Age_Group': '20-25',
    'Credit_Score_Range': '>500',
    'Income': 75,
    'Loan_Amount': 15000,
    'Loan_to_Income_Ratio': 0.2,
    'High_Risk_Loan': 0
}

# Encode and predict
df = pd.DataFrame([new_applicant])
for col, le in label_encoders.items():
    if col in df.columns:
        df[col] = le.transform(df[col])

prediction = model.predict(df)
probability = model.predict_proba(df)[0][1]

print(f"Prediction: {'DEFAULT' if prediction[0] == 1 else 'NO DEFAULT'}")
print(f"Default Probability: {probability*100:.2f}%")
```

---

## Models Implemented

| Model | Description | Use Case |
|-------|-------------|----------|
| **Naive Bayes** | Probabilistic classifier based on Bayes' theorem | Baseline model, fast predictions |
| **Logistic Regression** | Linear model for binary classification | Interpretable, good baseline |
| **Decision Tree** | Tree-based model with interpretable rules | Feature importance, business rules |
| **Random Forest** | Ensemble of decision trees | **BEST MODEL** - High accuracy, robust |
| **Support Vector Machine** | Finds optimal hyperplane for classification | Non-linear patterns |
| **K-Nearest Neighbors** | Instance-based learning | Local patterns |

---

## Results

### Model Performance Comparison

| Model | Test Accuracy | Precision | Recall | F1 Score |
|-------|--------------|-----------|--------|----------|
| Naive Bayes | ~52% | ~68% | ~73% | ~70% |
| Logistic Regression | ~58% | ~70% | ~75% | ~72% |
| Decision Tree | ~65% | ~74% | ~78% | ~76% |
| **Random Forest** | **~70%** | **~77%** | **~82%** | **~79%** |
| SVM | ~61% | ~72% | ~76% | ~74% |
| KNN | ~63% | ~73% | ~77% | ~75% |

*Note: Exact results may vary slightly due to random state*

### Best Model: Random Forest Classifier

- **Test Accuracy**: 70%
- **Precision**: 77%
- **Recall**: 82%
- **F1 Score**: 79%

**Why Random Forest?**
- Best balance between precision and recall
- Handles complex feature interactions
- Provides feature importance insights
- Robust to overfitting
- Excellent for imbalanced datasets

### Feature Importance (Top 5)

1. **Credit_Score_Range** - Most critical factor
2. **Loan_to_Income_Ratio** - Business rule validation
3. **Loan_Amount** - High loan amounts increase risk
4. **Income** - Lower income = higher risk
5. **Loan_Purpose** - Personal/Other loans are riskier

---

## Business Impact

### Current State vs ML Model

| Metric | Business Rules | ML Model (Random Forest) | Improvement |
|--------|---------------|-------------------------|-------------|
| Approval Rate | 1.2% | ~30% | +2,400% |
| Precision | 66% | 77% | +17% |
| Recall | Low | 82% | Significant |
| F1 Score | N/A | 79% | Data-driven decisions |

### Financial Impact

1. **Reduced Defaults**: Identify 82% of potential defaulters
2. **Increased Approvals**: Approve more low-risk borrowers
3. **Better Risk Management**: Consistent, objective decisions
4. **Cost Savings**: Fewer bad loans, reduced losses
5. **Revenue Growth**: More profitable lending opportunities

### Business Recommendations

1. **Deploy Random Forest model** for loan approval decisions
2. **Set risk thresholds** based on business tolerance
3. **Combine ML with business rules** for final approval
4. **Monitor model performance** continuously
5. **Retrain quarterly** with new data

---

## Key Findings

### High-Risk Indicators

1. **Credit Score 300-500**: 85.23% default rate
2. **Personal Loans**: 69.28% default rate
3. **Loan > 5x Income**: Significant risk factor
4. **Young Borrowers (20-25)**: Slightly higher risk
5. **Renters**: Higher default rate than homeowners

### Low-Risk Indicators

1. **Credit Score >500**: Lower default rate
2. **Medical Loans**: Lower default rate
3. **Loan < 5x Income**: Manageable debt
4. **Higher Income**: Better repayment capacity
5. **Homeowners**: More stable, lower risk

### Surprising Insights

- Gender has minimal impact on default rate
- Loan term (36 vs 60 months) shows small difference
- Income-to-loan ratio is more important than absolute income
- Multiple factors interact - single rules are insufficient

---

## Future Improvements

### Model Enhancements

1. **Hyperparameter Tuning**: GridSearchCV, RandomizedSearchCV
2. **Ensemble Methods**: Stacking, boosting (XGBoost, LightGBM)
3. **Cost-Sensitive Learning**: Assign different costs to FP vs FN
4. **Class Imbalance Handling**: SMOTE, class weights
5. **Deep Learning**: Neural networks for complex patterns

### Feature Engineering

1. **Payment History**: If available
2. **Debt-to-Income Ratio**: Total debt vs income
3. **Employment Status**: Job stability indicator
4. **Geographic Data**: Location-based risk
5. **Credit Utilization**: How much credit is used

### Deployment

1. **REST API**: Flask/FastAPI for predictions
2. **Real-time Scoring**: Production deployment
3. **A/B Testing**: Compare with current process
4. **Model Monitoring**: Track performance drift
5. **Automated Retraining**: Pipeline for model updates

---

## Technologies Used

- **Python 3.8+**
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - Machine learning models
- **Jupyter Notebook** - Interactive analysis

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Contact

For questions, feedback, or collaboration:

- **Project Lead**: [Your Name]
- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)

---

## Acknowledgments

- PrimeEdge Lending for providing the dataset
- Scikit-learn documentation and community
- Machine learning best practices from industry experts

---

**Made with Python and Machine Learning**
