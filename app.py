from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
rf_clf = joblib.load("random_forest_credit_risk_default.pkl")

# EXACT columns used during training (ORDER DOES NOT MATTER, NAMES DO)
FEATURES = [
    'person_age',
    'person_gender',
    'person_income',
    'person_emp_exp',
    'loan_amnt',
    'loan_int_rate',
    'loan_percent_income',
    'cb_person_cred_hist_length',
    'credit_score',
    'previous_loan_defaults_on_file',

    'person_education_Associate',
    'person_education_Bachelor',
    'person_education_Doctorate',
    'person_education_High School',
    'person_education_Master',

    'person_home_ownership_MORTGAGE',
    'person_home_ownership_OTHER',
    'person_home_ownership_OWN',
    'person_home_ownership_RENT',

    'loan_intent_DEBTCONSOLIDATION',
    'loan_intent_EDUCATION',
    'loan_intent_HOMEIMPROVEMENT',
    'loan_intent_MEDICAL',
    'loan_intent_PERSONAL',
    'loan_intent_VENTURE'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    # Initialize all features with 0
    data = {feature: 0 for feature in FEATURES}

    # -------- NUMERICAL INPUTS --------
    data['person_age'] = int(request.form['person_age'])
    data['person_gender'] = int(request.form['person_gender'])
    data['person_income'] = float(request.form['person_income'])
    data['person_emp_exp'] = int(request.form['person_emp_exp'])
    data['loan_amnt'] = float(request.form['loan_amnt'])
    data['loan_int_rate'] = float(request.form['loan_int_rate'])
    data['loan_percent_income'] = float(request.form['loan_percent_income'])
    data['cb_person_cred_hist_length'] = float(request.form['credit_history'])
    data['credit_score'] = int(request.form['credit_score'])
    data['previous_loan_defaults_on_file'] = int(request.form['previous_default'])

    # -------- EDUCATION (ONE HOT) --------
    edu = request.form['education']
    data[f'person_education_{edu}'] = 1

    # -------- HOME OWNERSHIP (ONE HOT) --------
    home = request.form['home_ownership']
    data[f'person_home_ownership_{home}'] = 1

    # -------- LOAN INTENT (ONE HOT) --------
    intent = request.form['loan_intent']
    data[f'loan_intent_{intent}'] = 1

    # Convert to DataFrame
    input_df = pd.DataFrame([data])

    # Prediction
    prediction = rf_clf.predict(input_df)[0]

    result = "❌ HIGH DEFAULT RISK" if prediction == 1 else "✅ LOW DEFAULT RISK"

    return render_template('index.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
