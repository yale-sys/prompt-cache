import glob
import json

import numpy as np
from tqdm import tqdm

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score
)

dataset2metric = {
    "squad_v2": qa_f1_score,
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    # "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}


def main():
    dset_list = [
        #"squad_v2",
        "narrativeqa",
        "qasper",
        "multifieldqa_en",
        "hotpotqa",
        "2wikimqa",
        "musique",
        "gov_report",
        "qmsum",
        "multi_news",
        # "trec",
        "triviaqa",
        # "samsum",
        "passage_count",
        "passage_retrieval_en",
        # "lcc",
        # "repobench-p",
    ]

    model_list = [
        #"falcon",
        "llama",
        #"mpt"
    ]

    for dset in tqdm(dset_list):
        print(f"Dataset: {dset}------------------------------")
        for m in model_list:
            p = f"./results_13b/{m}-{dset}"
            with open(glob.glob(f"{p}/no_cache_*.json")[0], "r") as f:
                no_cache = [json.loads(line) for line in f]
                no_cache_score, nc_std = score(no_cache, dset)

            with open(glob.glob(f"{p}/with_cache_*.json")[0], "r") as f:
                with_cache = [json.loads(line) for line in f]
                with_cache_score, wc_std = score(with_cache, dset)

            print(f"{m}-{dset}: {no_cache_score:.2f} ({nc_std:.2f}) vs {with_cache_score:.2f} ({wc_std:.2f}) ")


def score(results, dataset_name):
    scores_list = []
    for result in results:
        response = result["response"].split('</s>')[0].split('<|endoftext|>')[0].split('<|im_end|>')[0]
        answers = result["answers"]

        score = 0.
        for answer in answers:
            score = max(score, dataset2metric[dataset_name](response, answer))

        scores_list.append(score)

    #print(scores_list)
    mean = np.mean(scores_list) * 100
    std = np.std(scores_list)

    return mean, std


if __name__ == "__main__":
    main()
