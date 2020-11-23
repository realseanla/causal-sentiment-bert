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

final_dict = {}

for item in tqdm(sentiment_rank_copy):
    file_name = item[1]
    sentiment = item[0]
    with open('data_extracted/abstract_accepted.json', 'r') as jfile:
        jdict = json.load(jfile)
        name = file_name[:file_name.rfind('.')]
        accepted = jdict[name]['accepted']
    with open('data_extracted/dataset_20201122_161404/' + file_name, 'r') as file:
        abstract = file.read().replace('\n', ' ')
    final_dict[file_name] = {'abstract': abstract, 'sentiment': sentiment, 'accepted': accepted}
with open('data_sentiment/dataset.json', 'w') as json_file:
    json.dump(final_dict, json_file)




