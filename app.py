import pandas as pd
import numpy as np
import joblib
from NaiveBayes import NaiveBayesClassifier
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, render_template

app = Flask(__name__)
model = joblib.load('models/naive_bayes.pkl')

def get_symptom_list():
    data = []
    with open('data/SymptomList.txt', 'r', encoding='utf-8') as file:
        data = [line.strip() for line in file]
    return data

def get_disease_info(disease_name):
    info = pd.read_csv('data/disease_description.csv')
    return info[info.Disease == disease_name].Symptom_Description.values[0]

@app.route('/', methods=['GET', 'POST'])
def main():
    symptom_list_display = get_symptom_list()
    diagnoses = None
    confidence = None
    input_symptoms = None

    if request.method == 'POST':
        #convert HTTP request to pandas series
        model_input = {}
        for s in symptom_list_display:
            key = s.replace(' ', '_')
            key = key[0].lower() + key[1:]
            model_input[key] = int(request.form.get(s))
        model_input = pd.DataFrame([model_input])

        #get model predictions
        prob_mat = model.predict_proba(model_input)
        confidence = round(model.prediction_confidence(prob_mat)[0]*100, 0)
        predictions = prob_mat.columns[np.argsort(prob_mat.iloc[0,:].values)[-1:-4:-1]].values
        #predictions = model.predict(model_input)

        diagnoses = [{'disease':i, 'description':get_disease_info(i), 'score':round(prob_mat[i][0], 2)} for i in predictions if prob_mat.loc[0,i] > 0.05]
        #diagnoses = [{'disease':predictions[0], 'description':get_disease_info(predictions[0]), 'score':0.9}]
        if len(diagnoses) == 0:
            diagnoses = -1

        input_symptoms = model_input.columns[model_input.iloc[0,:].values == 1]
        input_symptoms = [i.replace('_', ' ') for i in input_symptoms]

    return render_template('index.html', all_symptoms=symptom_list_display, diagnoses=diagnoses, confidence=confidence, input_symptoms=input_symptoms)

if __name__ == '__main__':
    app.run(debug=True, port=10000)