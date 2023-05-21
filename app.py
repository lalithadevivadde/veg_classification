import os
from flask import Flask, render_template, request
from flask_cors import cross_origin, CORS
import tensorflow.lite as lite
from utils.make_upload_dir import make_upload_dir
from tensorflow.keras.utils import load_img, save_img, img_to_array
from numpy import array, round
from utils.delete_file import delete_file


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

interpreter = lite.Interpreter(model_path=os.path.join("models", "vegetable_classification_model_vgg.tflite"))


def predict(test_image):
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # test_image = expand_dims(test_image, axis=0).astype(input_details["dtype"])
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
        app.config['UPLOAD_FOLDER'] = os.path.join('.', 'static', 'uploads')
        make_upload_dir(app.config['UPLOAD_FOLDER'])

        upload_file = request.files['fileinput']
        upload_filename = upload_file.filename
        upload_file_path = os.path.join(app.config['UPLOAD_FOLDER'], upload_filename)

        if upload_filename == "":
            raise Exception("Please upload an image")

        if not upload_file_path.endswith(".jpg"):
            raise Exception("Not a JPG image")

        upload_file.save(upload_file_path)

        test_image = load_img(upload_file_path, color_mode="rgb", target_size=(224, 224))
        test_image = img_to_array(test_image)
        test_image = array([test_image])

        display_image = os.path.join('.', 'static', 'output', 'display.jpg')
        save_img(path=display_image, x=test_image[0])

        test_image = test_image / 255.
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
