from flask import Flask, request, render_template
import numpy as np
import pickle

application = Flask(__name__)
app = application

ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method == "POST":
        data = [
            float(request.form.get('Temperature')),
            float(request.form.get('RH')),
            float(request.form.get('Ws')),
            float(request.form.get('Rain')),
            float(request.form.get('FFMC')),
            float(request.form.get('DMC')),
            float(request.form.get('ISI')),
            float(request.form.get('Classes')),
            float(request.form.get('Region'))
        ]

        new_data_scaled = standard_scaler.transform([data])
        result = ridge_model.predict(new_data_scaled)[0]

        return render_template('home.html', results=round(result,2))

    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
