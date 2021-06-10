from flask import Flask, request
from ruleset import Ruleset
from waitress import serve

RULESET_PATH = "tweet-ruleset.json"
ruleset = Ruleset()
ruleset.load(RULESET_PATH)

app = Flask(__name__)


@app.route("/", methods=["POST"])
def predict():
    if request.form["text"]:
        return ruleset.predict(request.form["text"]).to_dict()
    else:
        return "no text input"


if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=9000)
