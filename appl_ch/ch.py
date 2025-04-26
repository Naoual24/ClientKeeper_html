from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Charger le modèle
model = joblib.load(open("ch.joblib", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    try:
        # Extraction des données du formulaire
        CreditScore = int(data["CreditScore"])
        Geography = data["Geography"]
        Age = int(data["Age"])
        Tenure = int(data["Tenure"])
        Balance = float(data["Balance"])
        NumOfProducts = int(data["NumOfProducts"])
        HasCrCard = int(data["HasCrCard"])
        IsActiveMember = int(data["IsActiveMember"])
        EstimatedSalary = float(data["EstimatedSalary"])
        Gender = data["Gender"]

        # Encodage one-hot Geography
        Geography_France = 1 if Geography == "France" else 0
        Geography_Germany = 1 if Geography == "Germany" else 0
        Geography_Spain = 1 if Geography == "Spain" else 0

        # Encodage one-hot Gender
        Gender_Female = 1 if Gender == 'Female' else 0
        Gender_Male = 1 if Gender == 'Male' else 0

        # Création du vecteur de caractéristiques
        features = np.array([[CreditScore, Age, Tenure, Balance, NumOfProducts,
                              HasCrCard, IsActiveMember, EstimatedSalary,
                              Geography_France, Geography_Germany, Geography_Spain,
                              Gender_Female, Gender_Male]])

        prediction = model.predict(features)[0]
        result = "Client à risque de départ" if prediction == 1 else "Client fidèle"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return f"Erreur lors de la prédiction : {e}"

if __name__ == '__main__':
    app.run(debug=True)
