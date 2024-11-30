
# LendWise

This Flask-based web application predicts loan approval using a machine learning model and explains the decision using SHAP values. It also stores each submission in a MySQL database for future reference.




## Features

- Loan Prediction: Predicts whether a loan application will be approved or rejected.
- SHAP-based Explanation: Provides feature importance and reasons for loan approval or rejection.
- Database Integration: Stores user input, prediction results, and SHAP explanations in a MySQL database.
- User-Friendly Interface: Interactive forms for submitting data and viewing results.


## Software

- Python 3.8+
- MySQL
- Flask
- Jupyter Notebook (for exploring SHAP values, optional)

## Libraries
Install required libraries using the command:

```
pip install flask flask_sqlalchemy pandas joblib shap mysql-connector-python scikit-learn
```
## Explanation Logic

### Loan Rejection Reasons:
The app uses SHAP values to identify features contributing negatively to the decision. Example reasons include:

- High number of dependents.
- Lower education level.
- Large loan amount.
- Low CIBIL score.
### Loan Approval Reasons:
Positive factors contributing to approval include:

- High CIBIL score.
- Sufficient income and assets.

## Sample Input
- ```no_of_dependents:``` Number of dependents.
- ```education:``` 0 (Not Graduate), 1 (Graduate).
- ```self_employed:``` 0 (No), 1 (Yes).
- ```age:``` Applicant's age in years.
- ```income_annum:``` Annual income in INR.
- ```loan_amount:``` Requested loan amount in INR.
- ```loan_term:``` Loan term in months.
- ```cibil_score:``` Applicant's CIBIL score.
- ```total_assets:``` Total assets owned in INR.
## Output
- Prediction: "Approved" or "Rejected".
- Probabilities: Likelihood of approval and rejection.
- Reasons: Key factors influencing the decision.
