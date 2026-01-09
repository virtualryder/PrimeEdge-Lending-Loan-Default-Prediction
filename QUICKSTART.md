# Quick Start Guide - PrimeEdge Lending Loan Default Prediction

## Getting Started in 5 Minutes

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run the Test Script (Optional)

Verify everything works:

```bash
python test_notebook.py
```

Expected output:
```
[SUCCESS] ALL TESTS PASSED - PIPELINE WORKING CORRECTLY!
Test Accuracy:  ~66%
Precision:      ~67%
Recall:         ~99%
F1 Score:       ~80%
```

### Step 3: Open the Jupyter Notebook

```bash
jupyter notebook PrimeEdge_Lending_Loan_Default_Prediction.ipynb
```

### Step 4: Run All Cells

In Jupyter: **Cell → Run All**

The notebook will:
1. Load and explore the dataset (11,548 loan applications)
2. Clean and preprocess data
3. Perform exploratory data analysis with visualizations
4. Train 6 ML models (Naive Bayes, Logistic Regression, Decision Tree, Random Forest, SVM, KNN)
5. Compare model performance
6. Identify Random Forest as the best model
7. Save the trained model for production use

### Step 5: Review Results

Check the model comparison section to see:
- **Best Model**: Random Forest
- **F1 Score**: ~80%
- **Recall**: ~99% (catches almost all defaults!)
- **Feature Importance**: Top risk factors identified

---

## Key Files

| File | Purpose |
|------|---------|
| `PrimeEdge_Lending_Loan_Default_Prediction.ipynb` | Main analysis notebook |
| `Loan_Delinquent_Analysis_Dataset.csv` | Dataset (11,548 records) |
| `test_notebook.py` | Quick test script |
| `requirements.txt` | Python dependencies |
| `README.md` | Full documentation |

---

## Expected Runtime

- **Test Script**: ~5 seconds
- **Full Notebook**: ~2-3 minutes (depending on your machine)

---

## Troubleshooting

### Issue: Module not found
**Solution**: Install requirements
```bash
pip install -r requirements.txt
```

### Issue: Jupyter not found
**Solution**: Install Jupyter
```bash
pip install jupyter
```

### Issue: Dataset not found
**Solution**: Ensure `Loan_Delinquent_Analysis_Dataset.csv` is in the same directory

---

## Next Steps

1. Review the **README.md** for detailed documentation
2. Explore the notebook to understand the analysis
3. Modify hyperparameters to improve model performance
4. Deploy the model using the saved `.pkl` files

---

## Quick Model Usage

```python
import pickle
import pandas as pd

# Load model
with open('best_loan_default_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make prediction
new_data = pd.DataFrame([{
    'Loan_Term': 0,  # Encoded value
    'Borrower_Gender': 1,
    'Loan_Purpose': 2,
    'Home_Status': 1,
    'Age_Group': 0,
    'Credit_Score_Range': 1,
    'Income': 75,
    'Loan_Amount': 15000,
    'Loan_to_Income_Ratio': 0.2,
    'High_Risk_Loan': 0
}])

prediction = model.predict(new_data)
print(f"Prediction: {'DEFAULT' if prediction[0] == 1 else 'NO DEFAULT'}")
```

---

**Happy Analyzing!**
