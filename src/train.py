
from sklearn.model_selection import train_test_split
from data import load_data, array_to_df
from src.text_analysis import TextCleaner
import gcld3


cleaner = TextCleaner()
cleaner.set_params(stem=False)
print(cleaner.stem)

RULESET_PATH = "tweet-ruleset.json"
#df = load_data("labelled-tweets-26-08-2021.json")
#x = array_to_df(df["words"])
#topics = array_to_df(df["topics"])
