import pickle
from sklearn.pipeline import Pipeline

save_folder = "models"


def load_model(role, label):
    filename = get_filename(role, label)
    folder = save_folder
    filepath = folder + "/" + filename
    model = None
    with open(filepath, 'rb') as file:
        model = pickle.load(file)
    return model


def load_pipe(role, label):
    return role, load_model(role, label)


def load_pipeline(label):
    vectorizer = load_pipe("vectorizer", label)
    reducer = load_pipe("reducer", label)
    model = load_pipe("model", label)
    return Pipeline([vectorizer, reducer, model])


def get_filename(role, topic):
    return role + "-" + topic + ".pkl"


class TweetLabeler:
    pipelines = []

    def __init__(self, labels):
        self.labels = labels

    def load(self):
        for label in self.labels:
            self.pipelines.append((label, load_pipeline(label)))

    def predict(self, text):
        labels = []
        for label, pipeline in self.pipelines:
            if pipeline.predict(text):
                labels.append(label)
        return labels
