# -*- coding: utf-8 -*-
"""
@author: Michael Akpevwe
"""
from flask import Flask, request
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.backend import set_session
from tensorflow.keras.models import load_model

app = Flask(__name__)

sess = tf.Session()
graph = tf.get_default_graph()

set_session(sess)
model = load_model('model.h5')

@app.route("/")
@app.route("/locationindex")
def index():
	return flask.render_template('locationIndex.html')

@app.route("/predict",methods = ['POST'])
def make_predictions():
    if request.method == 'POST':
        loc_month = request.args.get('month')
        loc_date = request.args.get('date')
        loc_hour = request.args.get('hour')
        loc_min = request.args.get('min')
        loc_sec = request.args.get('sec')
   
        #standardizing the input feature
        sc = StandardScaler()
        data = sc.fit_transform(np.array([[loc_month,loc_date,loc_hour,loc_min,loc_sec]]))
    
        global sess
        global graph
        with graph.as_default():
            set_session(sess)
            data = model.predict(data)
    
    return flask.render_template('predictPage.html' , response = data[0][0])
        
        
#@app.route('/predict_file', methods=['POST'])
#def predict_location_file():
#    
#    if request.method == 'POST':
#  
#        input_data =pd.read_csv(request.files.get('input_file'), header=None)
#    
#        #standardizing the input feature
#        sc = StandardScaler()
#        input_data = sc.fit_transform(input_data)
#   
#        global sess
#        global graph
#        with graph.as_default():
#            set_session(sess)
#            input_data = model.predict(input_data)
#        
#    return flask.render_template('predictPage.html' , response = str(list(input_data)))
#    
    

if __name__ == '__main__':
    app.run()