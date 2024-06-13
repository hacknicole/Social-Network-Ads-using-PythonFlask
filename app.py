from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

data = pd.read_csv('Social_Network_Ads.csv')

with open('model.pkl','rb') as pickle_model:
    model = pickle.load(pickle_model)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        userid = request.form['userid']
        print(userid)

        age = request.form['age']
        print(age)

        salary = request.form['salary']
        print(salary)

        features = np.array([[userid, age, salary]])
        prediction = model.predict(features)
        target = prediction[0]

        if target == 1:
            result = "Yes"
        else:
            result = "No"

        return render_template('index.html', result = result)

if __name__=='__main__':
    app.run(debug=True)
