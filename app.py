import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load the trained model
model = load_model('movie_classifier_model.h5')

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def home():
    return render_template('page1.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Preprocess the uploaded image
        img = image.load_img(file_path, target_size=(400, 400, 3))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0

        # Make a prediction
        predictions = model.predict(img)
        prediction = np.argmax(predictions, axis=1)
        
        # Map the prediction to the corresponding genre (assuming you have a genre mapping)
        genre_mapping = {0: 'Action', 1: 'Comedy', 2: 'Drama', 3: 'Horror', 4: 'Romance'}
        predicted_genre = genre_mapping.get(prediction[0], 'Unknown')
        
        return render_template('page1.html', prediction=predicted_genre)

if __name__ == '__main__':
    app.run(debug=True)
