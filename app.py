from flask import Flask,render_template,request
import numpy as np
import joblib
model=joblib.load('iris_lr.joblib')

app=Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods='POST')
def predict():
    f=[float(x) for x in request.form.values()]
    features=[np.array(f)]
    prediction=model.predict(features)
    return render_template('index.html',pred=''.format(prediction))

if __name__=='__main__':
    app.run(debug=True)
    
