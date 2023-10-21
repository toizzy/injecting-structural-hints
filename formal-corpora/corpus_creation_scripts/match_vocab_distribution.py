import argparse
from datasets import load_dataset
import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default = None)
    parser.add_argument("--data-type", type=str, default = "wikitext")
    parser.add_argument("--data-name", type=str, default = "wikitext-103-raw-v1")
    args = parser.parse_args()

    dataset = load_dataset(args.data_type, args.data_name)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # TODO could do some kind of smoothing?
    distribution = {index: 0 for token, index in tokenizer.get_vocab().items()}
    for example in tqdm(dataset["train"]):
        ids = tokenizer(example["text"])["input_ids"]
        for id in ids:
            distribution[id] += 1
    json.dump(distribution, open(Path("vocab_distributions") / f"{args.tokenizer}_{args.data_type}.json", "w"))
    
    

if __name__ == "__main__":
    main()

    
