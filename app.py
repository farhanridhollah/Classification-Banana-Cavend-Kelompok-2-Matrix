from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/upload/'  # Folder untuk menyimpan file gambar
model = load_model('myModel.h5')  # Ubah 'myModel.h5' sesuai dengan nama file model Anda

@app.route('/')
def beranda():
    return render_template('home.html')

@app.route('/klasifikasi', methods=['GET', 'POST'])
def klasifikasi():
    if request.method == 'POST':
        file = request.files['image']
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocessing gambar
        img = Image.open(filepath)
        img = img.resize((224, 224)) 
        img = np.array(img)  # Ubah ukuran gambar sesuai kebutuhan model Anda
        img = img / 255
        img = np.expand_dims(img, axis=0)

        # Prediksi dengan model
        prediction = model.predict(img)
        labels = ['Busuk', 'Masak', 'Mentah'] 
        predicted_label = labels[np.argmax(prediction)]
        # Lakukan sesuatu dengan hasil prediksi

        return render_template('klasifikasi.html', predicted_label=predicted_label)

    return render_template('klasifikasi.html')

if __name__ == '__main__':
    app.run(debug=True)
