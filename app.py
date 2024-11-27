from flask import Flask, render_template, request, send_from_directory
import cv2
import keras
import numpy as np
from keras.models import load_model

model = load_model('my_cnn_model.h5')
COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

@app.route('/')
def man():
    return render_template('index.html')


@app.route('/home', methods=['POST'])
def home():
    global COUNT
    img = request.files['image']

    img.save('static/{}.jpg'.format(COUNT))    
    img_arr = cv2.imread('static/{}.jpg'.format(COUNT))

    img_arr = cv2.resize(img_arr, (255,255))
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 255,255,3)
    prediction = model.predict(img_arr)

    return render_template('prediction.html', data=prediction)


@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('static', "{}.jpg".format(COUNT-1))


if __name__ == '__main__':
    app.run(debug=True)



