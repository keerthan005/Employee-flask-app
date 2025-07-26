from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Replace with actual features from your dataset
            age = float(request.form['age'])
            distance = float(request.form['distance'])
            income = float(request.form['income'])
            job_level = int(request.form['job_level'])

            # Build input array (adjust order & fields to match training)
            input_data = np.array([[age, distance, income, job_level]])

            pred = model.predict(input_data)[0]
            prediction = 'Yes' if pred == 1 else 'No'
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('form.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
