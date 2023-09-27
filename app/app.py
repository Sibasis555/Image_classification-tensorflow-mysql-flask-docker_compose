from flask import Flask, request, render_template
import mysql.connector
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import time
import os
 
app = Flask(__name__)

connection = mysql.connector.connect(
    user='root', password='root', host='mysql', port="3306", database='bird_classification')
print("DB connected")


# Set the folder to store uploaded images
current_dir = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(current_dir, 'artifacts')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values
    return img_array

# Load the trained model
model = tf.keras.models.load_model('model.h5')
# Define the class labels
class_labels = ['King Vulture', 'Masked Lapwing', 'Peacock', 'Victoria Crowned Pigeon', 'Violet Turaco', 'Wilsons Bird of Paradise', 'Woodland Kingfisher']  # Replace with your class labels
 
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    img_path = None
    if request.method == 'POST':
        image = request.files['image']
        if image:
            input_shape = model.input_shape[1:3]  # Target input image size

            save_img= Image.open(image)
            time_path = time.time()
            img_path = os.path.join(app.config['UPLOAD_FOLDER'],f"{time_path}.jpg")
            save_img.save(img_path)

            # cursor = mysql.connection.cursor()
            cursor = connection.cursor()
            img_file=f"{time_path}.jpg"

            cursor.execute("INSERT INTO bird_prediction (image_file) VALUES (%s)",[img_file])

            # Preprocess the input image
            processed_img = preprocess_image(img_path, input_shape)
            # Perform prediction using your prediction function
            # Make predictions
            predictions = model.predict(processed_img)
            predicted_class_index = np.argmax(predictions[0])
            prediction = class_labels[predicted_class_index]
            cursor.execute("UPDATE bird_prediction SET predection = %s WHERE image_file = %s",[prediction, img_file])
            cursor.close()
            return render_template("index.html", uploaded= True, prediction = prediction, img_path = img_path)
    # fetch all data stored in db
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM bird_prediction")
    birds= cursor.fetchall()
    cursor.close()
    print(birds)
    return render_template('index.html', uploaded=False)
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000)


