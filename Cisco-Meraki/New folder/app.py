from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import flask as Flask
from itertools import chain

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'C:/Users/Acer/Desktop/Deployment-Deep-Learning-Model-master/m1.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')

def large(c):
    if max(c)==c[0]:
      x1="Bacterial Pneumonia"
    elif max(c)==c[1]:
      x1="COVID-19"
    elif max(c)==c[2]:
      x1="Normal"
    elif max(c)==c[3]:
      x1="Viral Pneumonia"
    return x1



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        #preds = model_predict(file_path, model)
#def model_predict(img_path, model):
        img = image.load_img(file_path, target_size=(224, 224))

        # Preprocessing the image
        x = image.img_to_array(img)
        # x = np.true_divide(x, 255)
        x = np.expand_dims(x, axis=0)

        # Be careful how your trained model deals with the input
        # otherwise, it won't make correct prediction!
        img_data = preprocess_input(x)

        classes = model.predict(img_data)
        b=list(chain.from_iterable(classes))
        #c= tuple(b)
        preds = large(b)
        return preds
        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
    #    pred_class = decode_predictions(preds[0], top=1)   # ImageNet Decode
    #    result = pred_class[0]              # Convert to string

    return None


if __name__ == '__main__':
    app.run(debug=True)
