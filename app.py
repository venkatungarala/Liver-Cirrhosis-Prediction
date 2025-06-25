from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained artifacts
model = joblib.load('model/rf_model.pkl')
scaler = joblib.load('model/scaler.pkl')

@app.route('/', methods=['GET'])
def home():
    """Render the input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    1. Gather inputs
    2. Scale and predict
    3. Determine advice based on prediction
    4. Render result.html with both prediction and advice
    """
    try:
        features = [
            float(request.form['Age']),
            float(request.form['Gender']),
            float(request.form['BMI']),
            float(request.form['AlcoholConsumption']),
            float(request.form['Smoking']),
            float(request.form['GeneticRisk']),
            float(request.form['PhysicalActivity']),
            float(request.form['Diabetes']),
            float(request.form['Hypertension']),
            float(request.form['LiverFunctionTest']),
        ]
        arr = np.array([features])
        arr_scaled = scaler.transform(arr)
        pred = model.predict(arr_scaled)[0]

        if pred == 1:
            prediction_text = "🩺 Cirrhosis Detected"
            advice = [
                "Schedule an appointment with a hepatologist as soon as possible.",
                "Avoid alcohol entirely and limit fatty foods.",
                "Follow a balanced diet rich in fruits, vegetables, and lean protein.",
                "Stay hydrated and maintain a healthy weight.",
                "Regularly monitor liver function tests as advised by your physician."
            ]
        else:
            prediction_text = "✅ No Cirrhosis Detected"
            advice = [
                "Maintain your healthy lifestyle and balanced diet.",
                "Limit alcohol intake and avoid smoking.",
                "Engage in moderate exercise at least 3–5 times a week.",
                "Continue routine checkups and liver panels annually.",
                "Stay informed about liver health and risk factors."
            ]

    except KeyError as ke:
        prediction_text = f"⚠️ Missing input field: {ke}"
        advice = []
    except ValueError:
        prediction_text = "⚠️ All fields must be valid numbers."
        advice = []
    except Exception as e:
        prediction_text = f"⚠️ Unexpected error: {e}"
        advice = []

    return render_template('result.html',
                           prediction_text=prediction_text,
                           advice=advice)

if __name__ == '__main__':
    app.run(debug=True)
