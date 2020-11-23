import csv
import heapq
import json
from tqdm import tqdm


with open('data_sentiment/file_sentiment.csv') as csv_in:
    reader = csv.reader(csv_in, delimiter=',')
    sentiment_rank = [(row[2], row[0]) for row in reader]
    header = sentiment_rank[0]
    sentiment_rank = [(float(sr[0]), sr[1]) for sr in sentiment_rank[1:]]
    sentiment_rank_copy = sentiment_rank.copy()
    heapq.heapify(sentiment_rank)
    # n_largest = heapq.nlargest(10, sentiment_rank)

with open('data_sentiment/ordered_sentiment.csv', 'w', newline='') as csv_out:
    writer = csv.writer(csv_out)
    writer.writerow(header)
    for _ in range(len(sentiment_rank)):
        writer.writerow(heapq.heappop(sentiment_rank))

with open('dataset.json', 'r') as j_file:
    dataset = json.load(j_file)
for item in tqdm(sentiment_rank_copy):
    file = item[1][:item[1].rfind('.')]
    dataset[file]['sentiment'] = item[0]
with open('dataset.json', 'w') as j_file:
    json.dump(dataset, j_file)


