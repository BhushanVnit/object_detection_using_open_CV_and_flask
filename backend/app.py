import cv2
import os
import numpy as np
import requests
from PIL import Image
import io
from flask import Flask, flash, request, redirect, url_for, render_template, Response
from werkzeug.utils import secure_filename
import base64

app = Flask(__name__)

UPLOAD_FOLDER = 'images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_cars(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (5, 5), 0)
    dilated = cv2.dilate(blur, np.ones((3, 3)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    # Adjust the file path for cars.xml
    current_directory = os.path.dirname(os.path.abspath(__file__))
    car_cascade_src = os.path.join(current_directory, 'cars.xml')

    # Load the cascade classifier
    car_cascade = cv2.CascadeClassifier(car_cascade_src)

    cars = car_cascade.detectMultiScale(closing, 1.1, 1)
    
    for (x, y, w, h) in cars:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return image, len(cars)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Perform object detection
            image = cv2.imread(filepath)
            detected_image, num_cars = detect_cars(image)
            # Encode image data as base64
            _, buffer = cv2.imencode('.jpg', detected_image)
            image_bytes = base64.b64encode(buffer).decode('utf-8')
            # Render template with results
            return render_template('results.html', image_data=image_bytes, num_cars=num_cars)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
