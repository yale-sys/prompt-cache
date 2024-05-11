import fire
import json

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
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}


def main(result_file):
    dataset_name = result_file.split('/')[-3].split('-')[1]
    with open(result_file, 'r') as f:
        # load line by line
        results = [json.loads(line) for line in f]
        total_score = 0.
        for result in results:
            response = result["response"]
            answers = result["answers"]

            score = 0.
            for answer in answers:
                score = max(score, dataset2metric[dataset_name](response, answer))

            total_score += score
        print(total_score / len(results) * 100)


if __name__ == '__main__':
    fire.Fire(main)
