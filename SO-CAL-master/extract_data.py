import os
import json
import time
from tqdm import tqdm
from nltk.tokenize import sent_tokenize


def extract_acl():
    dataset = "acl_2017"
    with open("data_source/acl_accepted.txt") as f:
        accepted_titles = f.readlines()
        accepted_titles = list(set([t.strip() for t in accepted_titles]))
    extracted_data = {}
    for split in ["dev", "test", "train"]:
        parsed_pdfs_path = os.path.join("data_source/" + dataset, split, "parsed_pdfs")
        for file in os.listdir(parsed_pdfs_path):
            json_path = os.path.join(parsed_pdfs_path, file)
            with open(json_path, "r") as _json:
                j_dict = json.load(_json)
                title = j_dict["metadata"]["title"]
                _id = j_dict["name"][:j_dict["name"].find(".")]
                abstract = j_dict["metadata"]["abstractText"]
                accepted = title in accepted_titles
                extracted_data["acl_" + _id] = {"abstract": abstract, "split": split, "accepted": accepted}
    return extracted_data


def extract_arxiv(categories=("ai", "cl", "lg"), splits=("dev", "test", "train")):
    extracted_data = {}
    n_accepted = 0
    for c in categories:
        dataset = "arxiv.cs." + c + "_2007-2017"
        for s in splits:
            reviews_path = os.path.join("data_source", dataset, s, "reviews")
            for r in tqdm(os.listdir(reviews_path)):
                file_id = r[:r.rfind(".")]
                json_path = os.path.join(reviews_path, r)
                with open(json_path, "r") as _json:
                    j_dict = json.load(_json)
                    abstract = clean_abstract(j_dict["abstract"])
                    accepted = j_dict["accepted"]
                    title = j_dict["title"]
                    if accepted: n_accepted += 1
                extracted_data["arxiv_" + c + "_" + file_id] = {"title": title,
                                                                "abstract": abstract,
                                                                "split": s,
                                                                "accepted": accepted}
    print(n_accepted)
    return extracted_data


def generate_abstract_files(_dict):
    results_path = 'abstracts'
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    for paper in _dict.items():
        file_name = paper[0]
        sentence_list = sent_tokenize(paper[1]["abstract"])
        with open(os.path.join(results_path, file_name + '.txt'), 'w') as fout:
            for s in sentence_list:
                fout.write(s + '\n')


def save_dict(_dict):
    with open('dataset.json', 'w') as j_file:
        json.dump(_dict, j_file)


def clean_abstract(_str, rm_cites=True, replace_p=True):
    if rm_cites:
        while True:
            cite_start = _str.find("\\cite{")
            cite_end = _str.find("}")
            if cite_start == -1: break
            if cite_start > cite_end:
                cite_end = _str[cite_start:].find("}") + len(_str[:cite_start])
            if cite_start == 0:
                _str = _str[_str[cite_end + 1:]]
            else:
                _str = _str[:cite_start] + _str[cite_end + 1:]
    if replace_p:
        percent_idx = _str.find("%")
        if percent_idx == -1: return _str
        _str = _str.replace("%", " percent")
    return _str


def main():
    extracted_data = extract_arxiv()
    # generate_abstract_files(extracted_data)
    save_dict(extracted_data)


if __name__ == '__main__':
    main()
