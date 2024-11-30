from flask import Flask, request, render_template
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import joblib
import shap

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://<username>:<password>@<host>/<database>'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

model = joblib.load('random_forest_model.pkl')
explainer = shap.TreeExplainer(model)

class LoanRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    no_of_dependents = db.Column(db.Integer)
    education = db.Column(db.Integer)
    self_employed = db.Column(db.Integer)
    age = db.Column(db.Integer)
    income_annum = db.Column(db.Integer)
    loan_amount = db.Column(db.Integer)
    loan_term = db.Column(db.Integer)
    cibil_score = db.Column(db.Integer)
    total_assets = db.Column(db.Integer)
    prediction = db.Column(db.String(10))
    prob_approved = db.Column(db.Float)
    prob_rejected = db.Column(db.Float)
    reasons = db.Column(db.Text)

with app.app_context():
    db.create_all()

def explain_rejection_based_on_shap(feature_importance, user_input):
    rejection_reasons = []

    for index, row in feature_importance.iterrows():
        feature = row['Feature']
        feature_value = row['Feature Value']
        shap_value = row['SHAP Value']
        
        if shap_value < 0:  
            if feature == 'no_of_dependents' and user_input['no_of_dependents'] > 1:
                rejection_reasons.append(f"High number of dependents ({user_input['no_of_dependents']}) reduced the approval chances.")
            elif feature == 'education' and user_input['education'] == 0:
                rejection_reasons.append("Lower education level negatively impacted the loan approval.")
            elif feature == 'loan_amount' and user_input['loan_amount'] > 161600:
                rejection_reasons.append(f"Loan amount ({user_input['loan_amount']}) is higher than expected, contributing to rejection.")
            elif feature == 'age' and user_input['age'] < 33:
                rejection_reasons.append(f"Young age ({user_input['age']}) contributed negatively to loan rejection.")
            elif feature == 'cibil_score' and user_input['cibil_score'] < 600:
                rejection_reasons.append(f"Low CIBIL score ({user_input['cibil_score']}) reduced approval chances.")
            else:
                rejection_reasons.append(f"{feature} with value {user_input[feature]} contributed negatively to the decision.")

    if not rejection_reasons:
        rejection_reasons.append("No significant negative contributions to rejection were detected.")

    return rejection_reasons

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = {
            'no_of_dependents': int(request.form['no_of_dependents']),
            'education': int(request.form['education']),
            'self_employed': int(request.form['self_employed']),
            'age': int(request.form['age']),
            'income_annum': int(request.form['income_annum']),
            'loan_amount': int(request.form['loan_amount']),
            'loan_term': int(request.form['loan_term']),
            'cibil_score': int(request.form['cibil_score']),
            'total_assets': int(request.form['total_assets'])
        }

        user_input_df = pd.DataFrame([user_input])

        shap_values_user = explainer.shap_values(user_input_df)
        shap_values_for_rejected = shap_values_user[0][:, 0]

        feature_importance_user = pd.DataFrame({
            'Feature': user_input_df.columns,          
            'Feature Value': user_input_df.values.flatten(),  
            'SHAP Value': shap_values_for_rejected
        })

        prediction = model.predict(user_input_df)[0]
        probabilities = model.predict_proba(user_input_df)[0]

        result = {
            'prediction': 'Approved' if prediction == 1 else 'Rejected',
            'prob_approved': probabilities[1],
            'prob_rejected': probabilities[0]
        }

        if prediction == 0:  # Loan Rejected
            rejection_reasons = explain_rejection_based_on_shap(feature_importance_user, user_input)
            result['reasons'] = rejection_reasons
        else:  # Loan Approved
            result['reasons'] = [
                "High CIBIL score contributed positively to approval." if user_input['cibil_score'] >= 600 else "",
                "Income level met the expected threshold for approval." if user_input['income_annum'] >= 619000 else "",
                "Total assets met the expected threshold for approval." if user_input['total_assets'] >= 735300 else ""
            ]
            result['reasons'] = [r for r in result['reasons'] if r] 

        record = LoanRecord(
            no_of_dependents=user_input['no_of_dependents'],
            education=user_input['education'],
            self_employed=user_input['self_employed'],
            age=user_input['age'],
            income_annum=user_input['income_annum'],
            loan_amount=user_input['loan_amount'],
            loan_term=user_input['loan_term'],
            cibil_score=user_input['cibil_score'],
            total_assets=user_input['total_assets'],
            prediction=result['prediction'],
            prob_approved=float(result['prob_approved']),
            prob_rejected=float(result['prob_rejected']),
            reasons=', '.join(result['reasons'])
        )
        db.session.add(record)
        db.session.commit()

        return render_template('result.html', result=result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
