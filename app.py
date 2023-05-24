import os
from flask import Flask, render_template, request
from flask_cors import cross_origin, CORS
import tensorflow.lite as lite
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from utils.make_upload_dir import make_upload_dir
from tensorflow.keras.utils import load_img, save_img, img_to_array
from numpy import array, round
from utils.delete_file import delete_file
from requests import get


app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

CLASS_INDICES = {'Bean': 0,
                 'Bitter_Gourd': 1,
                 'Bottle_Gourd': 2,
                 'Brinjal': 3,
                 'Broccoli': 4,
                 'Cabbage': 5,
                 'Capsicum': 6,
                 'Carrot': 7,
                 'Cauliflower': 8,
                 'Cucumber': 9,
                 'Papaya': 10,
                 'Potato': 11,
                 'Pumpkin': 12,
                 'Radish': 13,
                 'Tomato': 14}

interpreter = lite.Interpreter(model_path=os.path.join("models", "vegetable_classification_model_mnet.tflite"))


def predict(test_image):
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    test_image = test_image.astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], test_image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]

    prediction = [*CLASS_INDICES.keys()][output.argmax()]
    probability = round(output.max() * 100, 2)
    result = f'{prediction} ({probability}%)'
    return result


@app.route("/")
@cross_origin()
def index():
    """
    displays the index.html page
    """
    display_image = os.path.join("static", "white.png")
    delete_file(os.path.join('.', 'static', 'uploads'))
    delete_file(os.path.join('.', 'static', 'output', 'display.jpg'))
    return render_template("index.html", display_image=display_image)


@app.route("/", methods=["POST"])
@cross_origin()
def file_prediction():
    upload_file_path = ""
    result = ''
    error = ""
    try:
        upload_file_path = os.path.join('.', 'static', 'uploads')
        make_upload_dir(upload_file_path)
        upload_filename = "input.jpg"
        upload_file_path = os.path.join(upload_file_path, upload_filename)

        file_url = request.form['fileinput']
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'}
        r = get(file_url, headers=headers)
        with open(upload_file_path, "wb") as file:
            file.write(r.content)

        test_image = load_img(upload_file_path, color_mode="rgb", target_size=(224, 224))
        test_image = img_to_array(test_image)
        test_image = preprocess_input(test_image)
        test_image = array([test_image])

        display_image = os.path.join('.', 'static', 'output', 'display.jpg')
        save_img(path=display_image, x=test_image[0])

        result = predict(test_image)
        delete_file(upload_file_path)

    except Exception as e:
        # raise
        result = ''
        error = e
        display_image = os.path.join("static", "white.png")
        delete_file(upload_file_path)

    return render_template("index.html", result=result, error=error, display_image=display_image)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
