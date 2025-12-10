import flask
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import werkzeug.utils
import numpy as np
import joblib
import transformers


GENERATOR = transformers.pipeline("text-generation", model="gpt2")
MODEL_IMAGE_CLASSIFICATION = VGG16()
MODEL_REGRESSION = joblib.load("./models/model.joblib")

ALLOWED_EXTENSIONS = {"png", "jpeg", "jpg", "jpeg"}

app = flask.Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def main():
    return flask.render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    image_file = flask.request.files["imagefile"]
    if not check_validation(str(image_file.filename)):
        return flask.render_template("index.html")  # idk lol
    else:
        image_file_name = werkzeug.utils.secure_filename(str(image_file.filename))
        image_path = "./src/images/" + image_file_name
        image_file.save(image_path)
        prediction_string = classify_image(image_path, model=MODEL_IMAGE_CLASSIFICATION)
        return flask.render_template(
            "index.html",
            prediction=prediction_string,
            image_path=f"/images/{image_file_name}",
        )


def check_validation(image_name: str):
    first_check = "." in image_name
    second_check = image_name.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    return first_check & second_check


def classify_image(image_path, model):
    image_loaded = load_img(image_path, target_size=(224, 224))
    image_array = img_to_array(image_loaded)
    image_array_batched = image_array.reshape(
        (1, image_array.shape[0], image_array.shape[1], image_array.shape[2])
    )
    image_array_preprosssed = preprocess_input(image_array_batched)
    prediction = model.predict(image_array_preprosssed)
    label_decoded = decode_predictions(prediction)
    label = label_decoded[0][0]
    return f"Object : {label[1]}, Confidence : {float(label[2]) * 100 :.2f}"


@app.route("/regpredict", methods=["POST"])
def predict_regression(model=MODEL_REGRESSION):
    int_features = [int(x) for x in flask.request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return flask.render_template(
        "index.html", prediction_text="Employee Salary should be $ {}".format(output)
    )


@app.route("/images/<filename>")
def request_file(filename):
    full_images_dir = os.path.join(os.path.dirname(__file__), "images")
    return flask.send_from_directory(full_images_dir, filename)


@app.route("/textgen", methods=["POST"])
def chat_gpt():
    # TODO : make it behave like a chat, actual chat
    prompt = flask.request.form["user_prompt"]
    generated_text = GENERATOR(prompt, max_length=300)[0]["generated_text"]
    return flask.render_template("index.html", generated_text=generated_text)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
