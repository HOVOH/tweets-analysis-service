from flask import Flask, request
from ruleset import Ruleset
from waitress import serve
from tweet_labeler import TweetLabeler
RULESET_PATH = "tweet-ruleset.json"
ruleset = Ruleset()
ruleset.load(RULESET_PATH)

app = Flask(__name__)
tweet_labeler = TweetLabeler(["crypto", "NFT", "defi"])
tweet_labeler.load()

@app.route("/tweet/label", methods=["POST"])
def predict():
    if request.form["text"]:
        return {"labels": tweet_labeler.predict([request.form["text"]])}
    else:
        return "no text input"


if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=9000)
