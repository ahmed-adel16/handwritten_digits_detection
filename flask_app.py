import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.neural_network import MLPClassifier
import joblib
from PIL import Image

app = Flask(__name__, template_folder='templates')

model = joblib.load('mlp_model.pkl')  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        img_file = request.files['file']
        img = Image.open(img_file.stream).convert('L').resize((28, 28))

        img_array = np.array(img).reshape(1, 28*28) / 255.0

        prediction = model.predict(img_array)

        predicted_digit = int(prediction[0])

        return jsonify({'predicted_digit': predicted_digit})
    except Exception as e:
        return jsonify({'error': str(e)})
import os
print("Current working directory:", os.getcwd())

if __name__ == "__main__":
    app.run(debug=True, port = 5050)