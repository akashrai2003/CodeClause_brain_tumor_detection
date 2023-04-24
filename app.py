import os
import tensorflow as tf
import numpy as np
from scipy.special import expit
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename


app = Flask(__name__)


model =load_model('C:/Users/akash/Downloads/BrainTumor Classification DL/Brain_Tumor_with_binarycrossentropy.h5')
print('Model loaded. Check http://127.0.0.1:5000/')


def get_className(classNo):
	if classNo==0:
		return "No Brain Tumor"
	elif classNo==1:
		return "Yes has Brain Tumor"


def getResult(img):
    img=cv2.imread(img)
    img = Image.fromarray(img)
    img = img.resize((64, 64))
    iiimage=np.array(img) # noinspection PyTypeChecker
    image = iiimage
    input_img = np.expand_dims(image, axis=0)
    result=model.predict(input_img)
    return result


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value=getResult(file_path)
        result=get_className(value) 
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)