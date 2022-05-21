from drqa import retriever
import random
from tqdm import tqdm
import jsonlines
import json
import click
import os
import time

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

@click.command()
@click.option("--model", type=click.Path(exists=True), help="Path to .npz tfidf model")
@click.option("--fills", type=click.Path(exists=True), help="jsonl file containing fills")
@click.option("--clues", type=click.Path(exists=True), help="jsonl file containing clues")
@click.option("--out", type=click.Path(exists=True), help="directory to put output in")
@click.option("--k", type=int, default=100, help="number of closest docs to search")
@click.option("--len-filter/--no-len-filter", default=True, help="Don't use length filter with negative matching")
def main(model, fills, clues, out, len_filter, k):
    os.makedirs(model, exist_ok=True)
    model = os.path.join(model, os.listdir(model)[0])
    ranker = retriever.get_class('tfidf')(tfidf_path=model)
    clues_ = []
    fills_ = []
    with jsonlines.open(clues) as reader:
        for line in reader:
            clues_.append(line["text"])

    with jsonlines.open(fills) as reader:
        for line in reader:
            fills_.append(line["text"])

    pairs = list(zip(clues_, fills_))

    top_negative = []
    print("Finding negative examples")
    for b in tqdm(list(batch(pairs, n=1000))):
        clues, fills = list(zip(*b))
        try:
            batch_doc_names, batch_doc_scores = list(zip(*ranker.batch_closest_docs(clues, k))) 
            j = 0
            for i in range(len(clues)):
                clue, fill = clues[i], fills[i]
                doc_ids = [int(doc_name[3:]) for doc_name in batch_doc_names[i]]
                found = False
                for i in doc_ids:
                    pair = pairs[i]
                    if pair[0] != clue and pair[1] != fill:
                        top_negative.append(pair)
                        found = True
                        break
                if not found:
                    while True:
                        cand = random.choice(pairs)
                        if not len_filter or len(cand[1]) == len(fill):
                            top_negative.append(cand)
                            break
                j +=1
        except Exception as e:
            for c, fill in zip(clues, fills):
                while True:
                    cand = random.choice(pairs)
                    if len(cand[1]) == len(fill):
                        top_negative.append(cand)
                        break
    print("Building json")
    filename = os.path.join(out, "train.json")
    build_json(pairs, top_negative, filename)

def build_json(pos_examples, neg_examples, filename):
    """ Takes in two equal length lists pos_examples and neg_examples, and builds the json for dpr.
    """
    assert len(pos_examples) == len(neg_examples)
    print(f"Building json... at {filename}")
    joined_ex = []
    for i in tqdm(range(len(pos_examples))):
        example = {}
        example["question"] = pos_examples[i][0]
        example["answers"] = [],
        example["positive_ctxs"] = [{
            "text": pos_examples[i][1],
            "title": ""}]
        example["hard_negative_ctxs"] = [{
            "text": neg_examples[i][1],
            "title": ""}]
        example["negative_ctxs"] = []
        joined_ex.append(example)
    with open(filename, "w") as f:
        joined_json = json.dumps(joined_ex, indent=4)
        f.write(joined_json)

if __name__ == "__main__":
    main()