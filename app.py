import pickle
import numpy as np
from math import log10
from flask import Flask
from flask import request
from flask import jsonify

class Perceptron():
    
    def __init__(self,eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        self.w_ = np.zeros(1+X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update = self.eta*(target-self.predict(xi))
                self.w_[1:] += update*xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:])+self.w_[0]
    
    def predict(self, X):
        return np.where(self.net_input(X)>=0.0,1,-1)

# Tworzenie instancji Flask
app = Flask(__name__)

# Tworzenie API
@app.route('/api/v1.0/predict', methods=['GET'])
def get_prediction():

    sepal_length = float(request.args.get('sl'))
    sepal_width = float(request.args.get('sw'))
    petal_length = float(request.args.get('pl'))
    petal_width = float(request.args.get('pw'))

    features = [sepal_length, sepal_width, petal_length, petal_width]

    # Załadowanie wygenerowanego modelu
    with open('irys.pkl',"rb") as picklefile:
        model = pickle.load(picklefile)

    # Predykcja
    predicted_class = int(model.predict(features))
    
    # Zwracanie wyniku
    jsn = jsonify(features=features, predicted_class=predicted_class)
    return f'<h2 style="color:blue">Cechy: {features}<br>Rodzaj kwiatuszka: {predicted_class}</h2> <img src="https://gifyagusi.pl/img/2018/07/02/obrazek-gif/milego-dnia/dzie%C5%84%20dobry%20mi%C5%82ego%20dnia%20smacznej%20kawusi.gif">'

if __name__ == '__main__':
    app.run(port=5005)
