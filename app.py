from flask import Flask, render_template, request
from disease_prediction import predict_disease

app = Flask(__name__)

# Home page with a form for symptoms input
@app.route('/')
def home():
    return render_template('index.html')

# Take data and analyze
@app.route('/symptoms/add', methods = ['POST'])
def predict():
    symptoms = request.form.getlist('symptoms')
    predicted_disease = predict_disease(symptoms)
    cleaned_predicted_disease = str(predicted_disease[0])
    return render_template('result.html', symptoms = symptoms, predicted_disease=cleaned_predicted_disease)

if __name__ == '__main__':
    app.run(debug=True)