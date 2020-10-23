# -*- coding: utf-8 -*-

import base64
import numpy as np
from keras.models import load_model
from keras.backend import set_session
from keras.preprocessing.image import img_to_array
from PIL import Image
from flask import request
from flask import jsonify
from flask import Flask
import io
from flask import render_template
import tensorflow as tf
import json

app = Flask(__name__)

def get_model():
    global model
    model = load_model('bird_image_classifier.h5')
    global graph
    graph = tf.get_default_graph()
    print('model loaded')
    

def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((112,112))
    image = img_to_array(image)
    image = np.expand_dims(image,axis=0)
    return image

sess=tf.Session()
print("loading model")
set_session(sess)
get_model()

@app.route("/predict",methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    print(decoded)
    image = Image.open(io.BytesIO(decoded))
    preprocessed_image = preprocess_image(image)
    preprocessed_image = preprocessed_image/255.0
    print(preprocessed_image)
    with graph.as_default():
        set_session(sess)
        prediction = model.predict(preprocessed_image)
        classes = model.predict_classes(preprocessed_image)
    response = {
                    "Fox Sparrow" : prediction[0][0],
                    "Gray Catbird": prediction[0][1],
                    "green kingfisher": prediction[0][2],
                    "indigo bunting": prediction[0][3],
                    "purple finch": prediction[0][4],
                    "rufous hummingbird": prediction[0][5],
                    "black footed albatross": prediction[0][6],
                    "cliff swallow": prediction[0][7],
                    "red headed woodpecker": prediction[0][8],
                    "white pelican": prediction[0][9]
            }
    print(response)
    print(type(response))
    return json.dumps(str(response))

@app.route("/")
def home():
    return render_template('predict.html')
if __name__ == '__main__':
    app.debug=True
    app.run()