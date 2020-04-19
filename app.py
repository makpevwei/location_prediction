# -*- coding: utf-8 -*-
"""
@author: Michael Akpevwe
"""
import flask
import json
import numpy as np
from sklearn.externals import joblib
from flask import Flask, render_template, request
from keras.models import model_from_json
import tensorflow as tf
from keras.backend import set_session
from keras.models import load_model

app = Flask(__name__)

sess = tf.Session()
graph = tf.get_default_graph()


@app.route("/")
@app.route("/locationindex")
def index():
	return flask.render_template('locationIndex.html')

@app.route("/predict",methods = ['POST'])
def make_predictions():
    if request.method == 'POST':
        a = request.form.get('month')
        b = request.form.get('date')
        c = request.form.get('hour')
        d = request.form.get('minute')
        e = request.form.get('second')
        
        X = np.array([[a,b,c,d,e]])
        global sess
        global graph
        with graph.as_default():
            set_session(sess)
            pred = loaded_model.predict(X)
        return flask.render_template('predictPage.html' , response = pred[0][0])
        
        
if __name__ == '__main__':
    set_session(sess)
    json_file = open("model.json","r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model.h5")

    app.run()