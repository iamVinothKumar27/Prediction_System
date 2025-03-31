from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load Loan Models
loan_model = pickle.load(open("model.pkl", "rb"))
loan_scaler = pickle.load(open("scaler.pkl", "rb"))

# Load Churn Model
churn_model = pickle.load(open("Customer_Churn_NoNumProducts.pkl", "rb"))

# Home page
@app.route('/')
def home():
    return render_template("index.html")

# Loan page
@app.route('/loan/')
def loan_form():
    return render_template("loan.html")

@app.route('/loan/predict', methods=['POST'])
def loan_predict():
    data = request.get_json()
    no_of_dep = int(data['no_of_dependents'])
    grad = data['education']
    self_emp = data['self_employed']
    annual_income = int(data['income_annum'])
    loan_amount = int(data['loan_amount'])
    loan_dur = int(data['loan_term'])
    cibil = int(data['cibil_score'])
    assets = int(data['assets'])

    grad_s = 0 if grad == 'Graduated' else 1
    emp_s = 0 if self_emp == 'No' else 1

    pred_data = pd.DataFrame([[no_of_dep, grad_s, emp_s, annual_income, loan_amount, loan_dur, cibil, assets]],
                             columns=['no_of_dependents', 'education', 'self_employed', 'income_annum',
                                      'loan_amount', 'loan_term', 'cibil_score', 'Assets'])

    pred_data = loan_scaler.transform(pred_data)
    prediction = loan_model.predict(pred_data)
    result = 'Loan Is Approved' if prediction[0] == 1 else 'Loan Is Rejected'

    return jsonify({'prediction': result})

# Churn page
@app.route('/churn/')
def churn_form():
    return render_template("churn.html")

@app.route('/churn/predict', methods=['POST'])
def churn_predict():
    data = request.get_json()  # Corrected to use get_json for JSON data
    features = [
        int(data['CreditScore']),
        1 if data['Gender'] == 'Male' else 0,
        int(data['Age']),
        float(data['Balance']),
        int(data['HasCrCard']),
        int(data['IsActiveMember']),
        float(data['EstimatedSalary'])
    ]
    final_input = np.array([features])
    prediction = churn_model.predict(final_input)
    result = "Customer is likely to churn." if prediction[0] == 1 else "Customer is likely to stay."

    return jsonify({'prediction_text': result})  # Return JSON response


if __name__ == '__main__':
    app.run(debug=True, port=5000)
