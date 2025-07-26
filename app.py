from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('form.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extracting input from form
        features = [
            float(request.form['Age']),
            float(request.form['BusinessTravel']),
            float(request.form['DailyRate']),
            float(request.form['Department']),
            float(request.form['DistanceFromHome']),
            float(request.form['Education']),
            float(request.form['EducationField']),
            float(request.form['Gender']),
            float(request.form['HourlyRate']),
            float(request.form['JobInvolvement']),
            float(request.form['JobLevel']),
            float(request.form['JobRole']),
            float(request.form['JobSatisfaction']),
            float(request.form['MaritalStatus']),
            float(request.form['MonthlyIncome']),
            float(request.form['NumCompaniesWorked']),
            float(request.form['OverTime']),
            float(request.form['PercentSalaryHike']),
            float(request.form['TotalWorkingYears']),
            float(request.form['YearsAtCompany']),
            float(request.form['YearsInCurrentRole']),
            float(request.form['YearsSinceLastPromotion']),
            float(request.form['YearsWithCurrManager'])
        ]

        final_input = np.array(features).reshape(1, -1)
        prediction = model.predict(final_input)[0]

        return render_template('form.html', prediction="Yes" if prediction == 1 else "No")

    except Exception as e:
        return f"❌ Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
