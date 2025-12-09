import flask

app = flask.Flask(__name__)


@app.route("/", methods=["GET"])
def main():
    return flask.render_template("index.html")


@app.route("/", methods=["POST"])
def predict():
    image_file = flask.request.files["imagefile"]
    image_path = "./images/" + str(image_file.filename)
    image_file.save(image_path)
    return flask.render_template("index.html")


if __name__ == "__main__":
    app.run(port=3000, debug=True)
