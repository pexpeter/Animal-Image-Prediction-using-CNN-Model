from flask import Flask, send_from_directory
import pandas as pd
import os

app = Flask(__name__)

# Animal species from the train generator
animal_classes = pd.read_csv('animaldict.csv')

# Serve the index.html file from the static directory
@app.route('/')
def index():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'index.html')

# Handle the form submission and serve the result page from the static directory
@app.route('/result', methods=['POST'])
def show_prediction():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'result.html')

if __name__ == '__main__':
    app.run()
