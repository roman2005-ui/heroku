from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd
from popularity import *



model = pickle.load(open('popular.pkl', 'rb'))
app = Flask(__name__,template_folder='template')

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def home():
    data1 = request.form['a']
    preds = recommend(data1)
    print(preds)
    return render_template('after.html',data=preds)


if __name__ == "__main__":
    app.run(debug=True,host='localhost',port=4000)