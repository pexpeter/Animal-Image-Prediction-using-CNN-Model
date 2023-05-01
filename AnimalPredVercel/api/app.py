from keras.preprocessing.image import  ImageDataGenerator
from keras.utils import img_to_array
from flask import Flask, request, render_template, redirect, url_for
import os
import numpy as np
import joblib
from PIL import Image
from werkzeug.utils import secure_filename
import pandas as pd

app = Flask(__name__)


# Define the path to the uploads folder
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


#Animal species from the train generator
animal_classes = pd.read_csv('animaldict.csv')

def preprocess_image(file_storage):
    '''Reads a file from user and edits image to required formats same as the model training.
    
    Par:
        file_strorage: str(path) File selection of the image to predict.
    
    Returns:
        img_batch: numpy array format of the image to prredict.
    
    '''
    # Load the image, resize  and conversion to numpy array
    img = Image.open(file_storage.stream).convert('RGB')
    img = img.resize((224,224))
    img_array = img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    #normalize the image
    img_batch/=255.0
    return img_batch

@app.route('/predict', methods=['POST'])
def make_prediction():
    '''Predicts the name of the animal picture uploaded.
    
    Returns:
        class_label (str): Predicted name of the species.
        image_path (str): The path of the image to be predicted.
    
    '''
    # Load the model
    model_path = os.path.join(os.getcwd(), 'models', 'model.pkl')
    model = joblib.load(model_path)
    
    # Get the image file from the request
    image = request.files['image']
    
    # Save the image to the upload directory with a unique filename
    image_filename = secure_filename(image.filename)
    image.save(os.path.join(app.config['UPLOAD_FOLDER'], image_filename))
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
    # Preprocess the image
    image = preprocess_image(image)
    
    # Make a prediction
    pred = model.predict(image)
    
    # Get the predicted class name
    class_idx = np.argmax(pred)
    class_label = animal_classes['index'][class_idx]
    
    # Redirect to the prediction result page
    return redirect(url_for('show_prediction', class_name=class_label, image_path=image_path))
    
    
@app.route('/result')
def show_prediction():
    class_name = request.args.get('class_name')
    image_path = request.args.get('image_path')
    # Render the prediction result page
    return render_template('result.html', prediction=class_name, image_path=image_path)
    
    
@app.route('/')
def index():
    animals = animal_classes['index'].tolist()
    return render_template('index.html', animals=animals)


if (__name__ == '__main__'):
    app.run(debug=True)

