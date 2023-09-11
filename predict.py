from flask import Flask, request, render_template
import your_prediction_function  # Replace with your actual prediction function

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    uploaded_image = None

    if request.method == 'POST':
        image = request.files['image']
        if image:
            # Perform prediction using your prediction function
            prediction = your_prediction_function(image)
            uploaded_image = image.filename

    return render_template('index.html', prediction=prediction, uploaded_image=uploaded_image)

if __name__ == '__main__':
    app.run(debug=True)
