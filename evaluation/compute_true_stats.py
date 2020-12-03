import argparse
from create_synthetic_dataset import load_json, save_json
import math
import pandas as pd
import sys


def data_to_dataframe(json: dict):
    # Return a dataframe with columns Y, T, Z
    # where Y is 'accepted', T is 'sentiment', and Z is 'confounder'
    Y, T, Z = [], [], []
    for paper_info in json.values():
        if paper_info['split'] == 'test':
            Y.append(paper_info['accepted'])
            T.append(paper_info['sentiment'])
            Z.append(paper_info['confounder'])
    return pd.DataFrame({'Y': Y, 'T': T, 'Z': Z})


def compute_marginal_prob(target_col: str, target_val: int, data: pd.DataFrame):
    target_count = len(data[data[target_col] == target_val])
    total_count  = len(data)
    return target_count/total_count


def compute_conditional_prob(target_col: str, condition_col: str, target_val: int, condition_val: int, data: pd.DataFrame):
    joint_count = len(data[(data[target_col] == target_val) & (data[condition_col] == condition_val)])
    marginal_count = len(data[data[condition_col] == condition_val])
    return joint_count/marginal_count


def compute_prob_y1(t: int, z: int, b: float, prop_score_val: float):
    # compute P(Y=1|T=t,Z=z): sigmoid(0.25*t+b*(P(T=1|Z=z)-0.2))
    logit = 0.25*t + b*(prop_score_val-0.2)
    prob = 1 / (1 + math.exp(-logit))
    return prob


def compute_ate(data: pd.DataFrame, b: float):
    # ATE: E_{p(Z)}[p(Y=1|T=1,Z=z)] - E_{p(Z)}[p(Y=1|T=0,Z=z)]
    prob_z0 = compute_marginal_prob(target_col='Z', target_val=0, data=data)
    prob_z1 = compute_marginal_prob(target_col='Z', target_val=1, data=data)

    prob_t1_z0 = compute_conditional_prob(target_col='T', condition_col='Z', target_val=1, condition_val=0, data=data)
    prob_t1_z1 = compute_conditional_prob(target_col='T', condition_col='Z', target_val=1, condition_val=1, data=data)

    prob_y1_do_t1 = prob_z0 * compute_prob_y1(t=1, z=0, b=b, prop_score_val=prob_t1_z0)\
                  + prob_z1 * compute_prob_y1(t=1, z=1, b=b, prop_score_val=prob_t1_z1)
    prob_y1_do_t0 = prob_z0 * compute_prob_y1(t=0, z=0, b=b, prop_score_val=prob_t1_z0)\
                  + prob_z1 * compute_prob_y1(t=0, z=1, b=b, prop_score_val=prob_t1_z1)
    
    return prob_y1_do_t1 - prob_y1_do_t0


def compute_att(data: pd.DataFrame, b: float):
    # ATT: E_{p(Z)}[p(Y=1|T=1,Z=z)] - E_{p(Z|T=1)}[p(Y=1|T=0,Z=z)]
    prob_z0 = compute_marginal_prob(target_col='Z', target_val=0, data=data)
    prob_z1 = compute_marginal_prob(target_col='Z', target_val=1, data=data)

    prob_t1_z0 = compute_conditional_prob(target_col='T', condition_col='Z', target_val=1, condition_val=0, data=data)
    prob_t1_z1 = compute_conditional_prob(target_col='T', condition_col='Z', target_val=1, condition_val=1, data=data)

    prob_z0_t1 = compute_conditional_prob(target_col='Z', condition_col='T', target_val=0, condition_val=1, data=data)
    prob_z1_t1 = compute_conditional_prob(target_col='Z', condition_col='T', target_val=1, condition_val=1, data=data)
    
    prob_y1_do_t1_t1 = prob_z0 * compute_prob_y1(t=1, z=0, b=b, prop_score_val=prob_t1_z0)\
                     + prob_z1 * compute_prob_y1(t=1, z=1, b=b, prop_score_val=prob_t1_z1)
    prob_y1_do_t0_t1 = prob_z0_t1 * compute_prob_y1(t=0, z=0, b=b, prop_score_val=prob_t1_z0)\
                     + prob_z1_t1 * compute_prob_y1(t=0, z=1, b=b, prop_score_val=prob_t1_z1)
    
    return prob_y1_do_t1_t1 - prob_y1_do_t0_t1
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', help='b in paper dataset', type=float, default=0.)
    parser.add_argument('-d', help='data json path',     type=str,   default='evaluation/synthetic/buzzyb0.json')
    parser.add_argument('-s', help='stats save path',    type=str,   default='evaluation/synthetic/buzzyb0_stats.json')
    args = parser.parse_args()

    dataset = load_json(args.d)
    df = data_to_dataframe(dataset)
    true_ate = compute_ate(data=df, b=args.b)
    true_att = compute_att(data=df, b=args.b)
    save_json(args.s, {'ATE': true_ate, 'ATT': true_att})
