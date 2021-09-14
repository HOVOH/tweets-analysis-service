import json
import pandas as pd
import numpy as np


def load_data(file):
    with open(file, 'r', encoding='utf8') as labelled_tweets_file:
        response = labelled_tweets_file.read()

    data = json.loads(response)
    results = data['results']
    texts = []
    topics = []
    for result in results:
        texts.append(result['text'])
        topics.append(result['meta']['topics'])

    df = pd.DataFrame(columns=["text", "topics"])
    df["text"] = texts
    df["topics"] = topics
    return df


def array_to_df(array):
    df = pd.DataFrame()
    for row in array:
        uniques = np.unique(row)
        df_row = pd.DataFrame(columns=uniques, data=[[True for i in uniques]])
        df = df.append(df_row, ignore_index=True)
    df = df.fillna(False)
    return df