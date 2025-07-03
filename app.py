from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend suitable for server use
import matplotlib.pyplot as plt

from uuid import uuid4

app = Flask(__name__)

model =load_model("ddd.keras")  # Make sure the path is correct
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

@app.route('/')
def home():
    return render_template('index.html')

from flask import Flask, request, render_template
import os
import matplotlib.pyplot as plt
from uuid import uuid4

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    prediction = None
    plot_path = None
    img_path = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Save uploaded file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            img_path = filepath

            # Preprocess and predict
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            prediction = model.predict(img_array)
            probs = prediction.flatten()

            # Get predicted class
            predicted_class = CLASS_NAMES[np.argmax(probs)]

            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(CLASS_NAMES, probs, color='skyblue')
            ax.set_ylabel('Confidence')
            ax.set_title('Driver Behavior Prediction Probabilities')
            plt.xticks(rotation=45, ha='right')
            plt.ylim([0, 1])
            plt.tight_layout()

            # Save plot
            plot_filename = f"{uuid4().hex}.png"
            plot_path_local = os.path.join("static", "plots", plot_filename)
            os.makedirs(os.path.dirname(plot_path_local), exist_ok=True)
            plt.savefig(plot_path_local)
            plt.close()

            # Convert to URL path for browser
            plot_path = f"/static/plots/{plot_filename}"

        return render_template(
            'upload.html',
            prediction=predicted_class,  # Only the top class
            img_path=img_path,
            plot_path=plot_path
        )

    return render_template('upload.html')
@app.route('/about')
def about():
    print(">>> DEBUG: /about route reached")
    return render_template('about.html')

@app.route('/aware')
def aware():
    return render_template('aware.html')

if __name__ == '__main__':
    app.run(debug=True)
