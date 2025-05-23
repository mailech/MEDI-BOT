from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import openai

app = Flask(__name__, static_folder='.', template_folder='.')
CORS(app)

medical_terms = {
    "myocardial infarction": "heart attack - when blood flow to part of the heart is blocked, causing damage to the heart muscle",
    "cerebrovascular accident": "stroke - when blood flow to part of the brain is interrupted, causing brain cells to die",
    "hypertension": "high blood pressure - when the force of blood against your artery walls is consistently too high",
    "hyperlipidemia": "high cholesterol - when you have too much of certain fats in your blood",
    "rhinitis": "runny or stuffy nose - often due to allergies or a cold",
    "otitis media": "middle ear infection - common in children",
    "gastroesophageal reflux disease": "GERD or acid reflux - when stomach acid frequently flows back into the tube connecting your mouth and stomach",
    "diabetes mellitus": "diabetes - a condition that affects how your body processes blood sugar",
    "osteoarthritis": "a type of arthritis that occurs when the protective cartilage that cushions the ends of your bones wears down over time",
    "chronic obstructive pulmonary disease": "COPD - a group of lung diseases that block airflow and make breathing difficult",
    "hypothyroidism": "underactive thyroid - when your thyroid gland doesn't produce enough of certain important hormones",
    "hyperglycemia": "high blood sugar - when there's too much sugar in your blood",
    "hypoglycemia": "low blood sugar - when there's not enough sugar in your blood",
    "neuralgia": "nerve pain - sharp, shooting, or burning pain due to irritated or damaged nerves",
    "pyrexia": "fever - when your body temperature is higher than normal",
    "dyspnea": "shortness of breath or difficulty breathing",
    "edema": "swelling caused by excess fluid trapped in your body's tissues",
    "arrhythmia": "irregular heartbeat - when your heart beats too fast, too slow, or irregularly",
    "hypotension": "low blood pressure - when the force of blood against your artery walls is consistently too low",
    "tachycardia": "fast heart rate - when your heart beats faster than normal",
    "bradycardia": "slow heart rate - when your heart beats slower than normal",
    "pulmonary arterial hypertension": "a type of high blood pressure that affects the arteries in your lungs and the right side of your heart - makes your heart work harder than normal to pump blood",
    "endothelial dysfunction": "damage to the thin layer of cells that line your blood vessels, causing them to not work properly",
    "vascular remodeling": "changes in the structure of blood vessels, often making them narrower",
    "in situ thrombosis": "blood clots that form and stay in one place",
    "mean pulmonary arterial pressure": "the average pressure in the arteries that carry blood from your heart to your lungs",
    "progressive disorder": "a condition that gets worse over time",
    "myalgia": "muscle pain or aches",
    "pruritis": "itchy skin",
    "syncope": "fainting or passing out briefly",
    "anemia": "a condition where you don't have enough healthy red blood cells to carry oxygen throughout your body",
    "vertigo": "a sensation of spinning or dizziness",
    "tinnitus": "ringing or buzzing noise in one or both ears",
    "dysphagia": "difficulty swallowing",
    "dysuria": "painful urination",
    "hematuria": "blood in urine",
    "hemoptysis": "coughing up blood",
    "epistaxis": "nosebleed",
    "paresthesia": "a burning or prickling sensation that is usually felt in the hands, arms, legs, or feet"
}

# Initialize Flask App
# app = Flask(__name__)

# Load datasets using relative paths
sym_des = pd.read_csv("C:/Users/SRIDHAR RAO/OneDrive/Desktop/AD-2/ad2/symtoms_df.csv")
precautions = pd.read_csv("C:/Users/SRIDHAR RAO/OneDrive/Desktop/AD-2/ad2/precautions_df.csv")
workout = pd.read_csv("C:/Users/SRIDHAR RAO/OneDrive/Desktop/AD-2/ad2/workout_df.csv")
description = pd.read_csv("C:/Users/SRIDHAR RAO/OneDrive/Desktop/AD-2/ad2/description.csv")
medications = pd.read_csv("C:/Users/SRIDHAR RAO/OneDrive/Desktop/AD-2/ad2/medications.csv")
diets = pd.read_csv("C:/Users/SRIDHAR RAO/OneDrive/Desktop/AD-2/ad2/diets.csv")
svc = joblib.load("C:/Users/SRIDHAR RAO/OneDrive/Desktop/AD-2/ad2/svc.pkl")

# Load trained model and label encoder
# svc = joblib.load("static/models/svc.pkl")
le = joblib.load("C:/Users/SRIDHAR RAO/OneDrive/Desktop/AD-2/ad2/label_encoder.pkl")  # Ensure LabelEncoder is saved during training

# Load symptoms dictionary from Training.csv column names
df = pd.read_csv("C:/Users/SRIDHAR RAO/OneDrive/Desktop/AD-2/ad2/Training.csv")
symptom_columns = df.columns[:-1]  # Exclude prognosis column
symptoms_dict = {symptom: idx for idx, symptom in enumerate(symptom_columns)}

# Mapping encoded labels to disease names
disease_classes = list(le.classes_)  # Get the disease names in correct order

# Helper function to fetch disease details
# def helper(dis):
#     desc = description.loc[description['Disease'] == dis, 'Description'].values[0]
#     pre = precautions.loc[precautions['Disease'] == dis, ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values[0]
#     med = medications.loc[medications['Disease'] == dis, 'Medication'].values[0]
#     die = diets.loc[diets['Disease'] == dis, 'Diet'].values[0]
#     wrkout = workout.loc[workout['disease'] == dis, 'workout'].values[0]
#     return desc, list(pre), med, die, wrkout

def helper(dis):
    desc = description.loc[description['Disease'] == dis, 'Description'].values[0]
    pre = precautions.loc[precautions['Disease'] == dis, ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values[0]
    
    med = medications.loc[medications['Disease'] == dis, 'Medication'].values[0].split(",")  # ✅ Convert string to list
    die = diets.loc[diets['Disease'] == dis, 'Diet'].values[0].split(",")  # ✅ Convert string to list
    wrkout = workout.loc[workout['disease'] == dis, 'workout'].values[0].split(",")  # ✅ Convert string to list
    
    return desc, list(pre), med, die, wrkout


# Function to predict disease based on symptoms
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))  # Ensure 132 features

    for item in patient_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1  # Set corresponding index to 1

    # Model prediction
    predicted_label = svc.predict([input_vector])[0]

    # Decode label back to disease name
    predicted_disease = le.inverse_transform([predicted_label])[0]
    
    return predicted_disease

# Routes
@app.route("/")
def index():
    return render_template("new.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        if not symptoms:
            return render_template('new.html', message="Please enter symptoms separated by commas.")

        user_symptoms = [s.strip() for s in symptoms.split(',')]
        predicted_disease = get_predicted_value(user_symptoms)
        
        if predicted_disease == "Unknown Disease":
            return render_template('new.html', message="Could not identify the disease.")

        dis_des, my_precautions, medications, my_diet, workout = helper(predicted_disease)

        return render_template('new.html', predicted_disease=predicted_disease, dis_des=dis_des,
                               my_precautions=my_precautions, medications=medications, my_diet=my_diet,
                               workout=workout)

    return render_template('new.html')

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact_copy')
def contact():
    return render_template("contact_copy.html")

@app.route('/blog')
def developer():
    return render_template("blog.html")

@app.route('/chat')
def chat():
    return render_template("chat.html")

@app.route('/simplify', methods=['POST'])
def simplify():
    data = request.get_json()
    input_text = data.get("text", "").lower().strip()

    if not input_text:
        return jsonify({"response": "Please enter a medical term or sentence."})

    found_terms = []
    for term, explanation in medical_terms.items():
        if term in input_text:
            found_terms.append({"term": term, "explanation": explanation})

    if "pulmonary arterial hypertension" in input_text:
        return jsonify({
            "response": "In simple terms: Pulmonary arterial hypertension (PAH) is a serious condition where the blood pressure in the lungs is too high. "
                        "This happens when the small blood vessels in the lungs become damaged over time. This damage causes the vessels to become narrow and sometimes blocked by small blood clots. "
                        "As this condition gets worse over time, it forces your heart to work much harder to pump blood through these narrowed vessels. The increased workload can eventually damage your heart. "
                        "Doctors can measure the pressure in these lung vessels to diagnose this condition."
        })

    if len(input_text) > 100 or ('.' in input_text and len(input_text.split('.')) > 1):
        return jsonify({"response": format_paragraph(input_text, found_terms)})

    if len(found_terms) == 1:
        t = found_terms[0]
        return jsonify({"response": f'"{t["term"]}" means: {t["explanation"]}'})
    elif len(found_terms) > 1:
        response = "I found multiple medical terms in your text:\n\n"
        for t in found_terms:
            response += f'• "{t["term"]}": {t["explanation"]}\n\n'
        return jsonify({"response": response})
    else:
        return jsonify({"response": "I couldn't identify specific medical terminology. Please provide a more detailed sentence or another medical term."})

def format_paragraph(text, found_terms):
    simplified = text
    for t in found_terms:
        simple = t['explanation'].split(" - ")[0].split(" or ")[0]
        simplified = simplified.replace(t['term'], simple)

    result = "Here's your medical text explained in simple terms:\n\n"
    result += "Simple version: " + simplified + "\n\n"
    result += "Medical terms explained:\n\n"
    for t in found_terms:
        result += f'• "{t["term"]}": {t["explanation"]}\n\n'
    return result

# Set your OpenAI API key (ensure you set this in your environment for security)
api_key = ""
if not api_key:
    raise ValueError("API key is required")
client = openai.OpenAI(api_key=api_key)

@app.route('/qa', methods=['POST'])
def qa():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"answer": "Please provide a question."}), 400
    try:
        # Updated OpenAI API call for version 1.0.0+
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": question}],
            max_tokens=256,
            temperature=0.7
        )
        answer = response.choices[0].message.content.strip()
        return jsonify({"answer": answer})
    except Exception as e:
        error_message = str(e)
        if "insufficient_quota" in error_message:
            return jsonify({
                "answer": "Sorry, the API quota has been exceeded. Please check your OpenAI account billing and quota status. You can:\n1. Add a payment method to your OpenAI account\n2. Check your usage limits\n3. Get a new API key"
            }), 429
        else:
            return jsonify({"answer": f"Error: {error_message}"}), 500

@app.route('/test_qa')
def test_qa():
    return render_template('test_qa.html')

if __name__ == '__main__':
    app.run(debug=True)
