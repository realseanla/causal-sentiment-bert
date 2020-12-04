import argparse
import copy
import json
import math
import numpy


"""
Run: python create_synthetic_dataset.py

TODO: Sentiment label must be 0 or 1!
"""

DATASET_PATH = 'SO-CAL-master/dataset.json'
BUZZY_WORDS  = ['deep','neural','embed','adversarial net']
numpy.random.seed(0)


def load_json(path):
    with open(path, 'r') as fp:
        return json.load(fp)

def save_json(path, content):
    with open(path, 'w') as fp:
        json.dump(content, fp)

def string_contains_words(string: str, words: list):
    for w in words:
        if w in string: return True
    return False


def batch_set_confounder_and_treatment(paper_entries: list):
    # Set confounder field to be true if title contains buzzy words
    # Set sentiment to 1 if sentiment score is positive else 0
    for paper in paper_entries:
        if string_contains_words(paper['title'], BUZZY_WORDS):
            paper['confounder'] = 1
        else:
            paper['confounder'] = 0
        if paper['sentiment'] > 0.:
            paper['sentiment'] = 1
        else:
            paper['sentiment'] = 0

def batch_compute_outcome_prob(paper_entries: list, b_1: float = 1.):
    # Compute propensity scores, then compute outcome probability
    num_conf_true, num_conf_false = 0, 0
    num_treated_conf_true, num_treated_conf_false = 0, 0
    for paper in paper_entries:
        confounder_value = paper['confounder']
        treatment_value  = paper['sentiment']
        if confounder_value == 1:
            num_conf_true += 1
            if treatment_value == 1:
                num_treated_conf_true += 1
        else:
            num_conf_false += 1
            if treatment_value == 1:
                num_treated_conf_false += 1
    propensity_scores = (num_treated_conf_true/num_conf_true, num_treated_conf_false/num_conf_false)

    probs = [0.] * len(paper_entries)
    for index, paper in enumerate(paper_entries):
        prop_score = propensity_scores[paper['confounder']]
        prob = 0.25*paper['sentiment'] + b_1*(prop_score-0.2)
        prob = 1 / (1 + math.exp(-prob))
        probs[index] = prob
    return probs

def batch_sample_bernoulli(probs: list):
    # Sample from Bernoulli based on probability
    thresholds = numpy.random.uniform(low=0.0, high=1.0, size=len(probs))
    samples    = [0] * len(probs)
    for index, prob in enumerate(probs):
        samples[index] = 1 if prob>thresholds[index] else 0
    return samples


def generate_synthetic_labels(dataset: dict, b_1: float = 1.):
    # Return a dictionary where true acceptance labels are replaced by synthetic acceptance labels
    synthetic_dataset = copy.deepcopy(dataset)
    keys, values = list(synthetic_dataset.keys()), list(synthetic_dataset.values())

    batch_set_confounder_and_treatment(values)
    probs   = batch_compute_outcome_prob(values, b_1)
    samples = batch_sample_bernoulli(probs)
    for index, content in enumerate(zip(keys, values)):
        key, value = content[0], content[1]
        value['accepted'] = samples[index]
        synthetic_dataset[key] = value
    return synthetic_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', help='b in paper dataset', type=float, default=1.)
    parser.add_argument('-s', help='save path',          type=str, default='evaluation/synthetic/buzzy.json')
    args = parser.parse_args()

    dataset = load_json(DATASET_PATH)
    synthetic_dataset = generate_synthetic_labels(dataset, b_1=args.b)
    save_json(args.s, synthetic_dataset)
