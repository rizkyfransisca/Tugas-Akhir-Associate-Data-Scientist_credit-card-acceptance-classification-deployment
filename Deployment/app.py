from flask import Flask, request, render_template,jsonify
import pickle
import numpy as np

app = Flask(__name__)

model_file = open('model.pkl', 'rb')
scaler_file = open('scaler.pkl', 'rb')

model = pickle.load(model_file, encoding='bytes')
scaler = pickle.load(scaler_file, encoding='bytes')

@app.route('/')
def index():
    return "Hello World"

@app.route('/predict', methods=['POST'])
def predict():
    json_data = request.json
    credit_rating, mailer_type, reward, income_level = json_data["credit_rating"], json_data["mailer_type"], json_data["reward"], json_data["income_level"]

    data = []
    if credit_rating == "High":
        data.append(2)
    elif credit_rating == "Medium":
        data.append(1)
    else:
        data.append(0)
    
    if mailer_type == "Letter":
        data.extend([1,0])
    else:
        data.extend([0,1])
    
    if reward == "Air Miles":
        data.extend([1,0])
    elif reward == "Cash Back":
        data.extend([0,1])
    else:
        data.extend([0,0])
    
    if credit_rating == "High":
        data.append(2)
    elif credit_rating == "Medium":
        data.append(1)
    else:
        data.append(0)

    data = scaler.transform([data])
    prediction = model.predict(data)
    if prediction == 1:
        prediction = "Yes"
    else:
        prediction = "No"
    return jsonify(
        prediction = prediction
    )

if __name__ == '__main__':
    app.run(debug=True)