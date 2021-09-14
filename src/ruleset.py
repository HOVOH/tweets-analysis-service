import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from text_analysis import get_words
import numpy as np

TEXT_PREFIX = "text:"
TOPIC_PREFIX = "topic:"


def sets_are_equal(set0, set1):
    if len(set0) != len(set1):
        return False
    for e in set0:
        if e not in set1:
            return False
    return True


class Ruleset:

    def fit(self, texts, topics):
        self.ruleset = pd.DataFrame(columns=["support", "topics", "keywords"])
        inputs = pd.DataFrame(columns=["text", "topics"])
        inputs.text = texts
        inputs.topics = topics

        df = pd.DataFrame(columns=["text", "topics"])
        for i, item in inputs.iterrows():
            df = df.append(
                {'text': get_words(item['text']), 'topics': item['topics']},
                ignore_index=True
            )

        print("--- Data Sample ---")
        print(df.head(10))

        encoded = pd.DataFrame()
        for i, row in df.iterrows():
            dict = {}
            for word in row.text:
                dict[TEXT_PREFIX + word] = True
            for topic in row.topics:
                dict[TOPIC_PREFIX + topic] = True
            encoded = encoded.append(dict, ignore_index=True)

        encoded.fillna(0, inplace=True)
        rules = fpgrowth(encoded, min_support=0.1, use_colnames=True)
        print("--- Extracted Rules Sample ---")
        print(rules.head(10))
        for i, rule in rules.iterrows():
            topics = []
            keywords = []
            prefixed_keywords = []
            for element in rule.itemsets:
                if element.startswith(TEXT_PREFIX):
                    prefixed_keywords.append(element)
                    word = element.replace(TEXT_PREFIX, "")
                    keywords.append(word)
                else:
                    topic = element.replace(TOPIC_PREFIX, "")
                    topics.append(topic)
            if len(topics) > 0 and len(keywords) > 0:
                equal_keywords = rules["itemsets"].apply(lambda t: sets_are_equal(t, prefixed_keywords))
                keyword_support = rules[equal_keywords]["support"].max()
                self.ruleset = self.ruleset.append({
                    "support": rule.support,
                    "confidence": rule.support/keyword_support,
                    "topics": topics,
                    "keywords": keywords},
                    ignore_index=True)

        print("--- Filter Rules ---")
        print(self.ruleset)

    def predict(self, text):
        words = get_words(text)
        filled_rules = []
        for i, rule in self.ruleset.iterrows():
            absent_keyword = [k for k in rule.keywords if not k in set(words)]
            if len(absent_keyword) == 0:
                filled_rules.append(i)

        if len(filled_rules) == 0:
            return pd.DataFrame(columns=["score"])

        topics_score = pd.DataFrame(columns=["topic", "score"])
        for filled_rule in filled_rules:
            rule = self.ruleset.loc[filled_rule]
            for topic in rule.topics:
                topics_score = topics_score.append({"topic": topic, "score": rule.support}, ignore_index=True)
        return topics_score.groupby("topic").max()

    def save(self, path):
        self.ruleset.to_json(path)

    def load(self, path):
        self.ruleset = pd.read_json(path)