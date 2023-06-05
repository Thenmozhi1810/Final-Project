import cv2
import os
import pathlib
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical

app = Flask(__name__)

model=load_model('EfficientLSTM.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

class_names = ['Mild_Demented','Moderate_Demented','Normal','Tumor','Tumor_mild','Tumor_verymild','Very_Mild_Demented']

def apply_filters(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.convertScaleAbs(img)
    img_pnlm = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    img_eq = cv2.equalizeHist(img_pnlm[:,:,0])
    img_out = np.stack([img_eq, img_eq, img_eq], axis=-1)
    return img_out

img_width, img_height = 224, 224
seq_length = 16

base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

def extract_features(image):
    img = load_img(image, target_size=(img_width, img_height))
    img = np.array(img)
    x = apply_filters(img)
    x = preprocess_input(x)
    features = base_model.predict(np.array([x]))
    return features

def getResult(img):
    features = extract_features(img)
    features = np.reshape(features, (features.shape[0], 16, -1))
    prediction = model.predict(features)
    prediction = np.array(prediction)
    predicted_class = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class]
    return predicted_class_name

@app.route('/')
@app.route('/home', methods =['GET', 'POST'])
def home():
    if request.method == 'POST':
        image = request.files.get('image')
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(image.filename))
        image.save(file_path)
        result = getResult(file_path)
        return render_template("result.html", y=result)

    return render_template("home.html")

if __name__ == '__main__':
    app.run(debug=True)