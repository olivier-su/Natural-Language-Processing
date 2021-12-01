import gensim.downloader as api
import pandas as pd

model = api.load("word2vec-google-news-300")
print(api.info("word2vec-google-news-300"))
synonyms = pd.read_csv('synonyms.csv')

