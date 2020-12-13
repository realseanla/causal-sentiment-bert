import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split

random.seed(0)
np.random.seed(0)

df = pd.read_json("reddit_sentiment_results.json").T
sentiment = pd.json_normalize(df.sentiment_all)
for column in sentiment:
    df.loc[:, column] = sentiment[column].to_numpy()

df.loc[:, 'sentiment'] = (df['pos'].to_numpy() > df['neg'].to_numpy()) & (df['pos'].to_numpy() > df['neu'].to_numpy())
df.loc[:, 'sentiment'] = df['sentiment'].astype(int)

df = df[['comment', 'score', 'sentiment']]

train, test = train_test_split(df, train_size=0.8, random_state=0)
test, dev = train_test_split(test, train_size=0.5, random_state=0)

df.loc[train.index, "split"] = "train"
df.loc[test.index, "split"] = "test"
df.loc[dev.index, "split"] = "dev"

df.to_csv('reddit_sentiment_processed.csv')
