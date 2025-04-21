from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)
model = load_model(r'ddd.keras')  # Make sure the path is correct
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CLASS_NAMES = ['Safe Driving', 'Texting - Right', 'Talking on the Phone - Right',
               'Texting - Left', 'Talking on the Phone - Left', 'Operating the Radio',
               'Drinking', 'Reaching Behind', 'Hair and Makeup', 'Talking to Passenger']

def model_predict(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    return CLASS_NAMES[np.argmax(preds)]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            prediction = model_predict(filepath)
            return render_template('index.html', prediction=prediction, img_path=filepath)
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
