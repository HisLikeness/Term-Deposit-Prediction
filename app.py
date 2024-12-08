from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model_path = r"c:\Users\HP\Documents\STUDY\DATA SCIENCE AND ANALYSIS - GOMYCODE\GMC_PYTHON\Checkpoints\DNN Checkpoint - Banking\classification.h5"
loaded_model = tf.keras.models.load_model(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    prediction = (loaded_model.predict(features) > 0.5).astype(int).tolist()
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
