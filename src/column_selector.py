class ColumnsSelector:

    def __init__(self, columns):
        self.columns = columns

    def fit(self, x, y):
        return self

    def transform(self, x):
        return x.loc[:, self.columns]


