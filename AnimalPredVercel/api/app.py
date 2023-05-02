from flask import Flask, request, render_template, redirect, url_for
import pandas as pd


app = Flask(__name__)

#Animal species from the train generator
animal_classes = pd.read_csv('animaldict.csv')

@app.route('/result', methods=['POST'])
def show_prediction():
    return render_template('result.html')
    
    
@app.route('/')
def index():
    animals = animal_classes['index'].tolist()
    return render_template('index.html', animals=animals)


if (__name__ == '__main__'):
    app.run(debug=True)

