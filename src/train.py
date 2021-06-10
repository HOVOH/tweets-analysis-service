import json

from ruleset import Ruleset

RULESET_PATH = "tweet-ruleset.json"

with open('labelled-tweets.json', 'r', encoding='utf8') as labelled_tweets_file:
    response = labelled_tweets_file.read()

data = json.loads(response)
results = data['results']

texts = []
topics = []
for result in results:
    texts.append(result['text'])
    topics.append(result['meta']['topics'])

ruleset = Ruleset()
ruleset.fit(texts, topics)
ruleset.save(RULESET_PATH)