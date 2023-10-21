import argparse
from collections import Counter
import json
import numpy as np
import os
from pathlib import Path
import pickle
from tqdm import tqdm

import sys
sys.path.append('..')

from utils import get_distribution

LINE_LENGTH = 512
# Corpus length is in tokens, number of lines computed in the code
CORPUS_LENGTHS = {"train": 1_000_000_000, "test": 1_000_000, "valid": 500_000}
MAX_CLOSING_DISPLACEMENT = 3

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default = 50_000)
    parser.add_argument("--vocab-distribution", type=str, choices = ["zipf", "uniform"])
    args = parser.parse_args()
    output_dir = "../../data/random"
    if args.vocab_size > 1_000:
        output_dir += f"_vocab{args.vocab_size // 1000}K"
    else:
        output_dir += f"_vocab{args.vocab_size}"
    output_dir += f"-{args.vocab_distribution}"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    word2idx = dict([(str(i), i) for i in range(args.vocab_size)])
    json.dump(word2idx, open(os.path.join(output_dir, f"vocab_limit{args.vocab_size}.json"), "w"))
    word_indices, vocab_ps = get_distribution(args.vocab_distribution, args.vocab_size)
    
    for split in ["valid", "train", "test"]:
        output_filename = os.path.join(output_dir, f"{split}.txt")
        print(f"On {split} split")
        split_length = CORPUS_LENGTHS[split]
        num_lines = split_length // LINE_LENGTH
        print(f"Going to make {num_lines} lines, for a corpus of {split_length} tokens")
        split_length = num_lines * LINE_LENGTH
        # Sampling all of the words in one go greatly increases speed. 
        # We're just flipping all coins in advance, does not change process.
        print("Frontloading all random samples, which might take a bit")
        vocab_samples = np.random.choice(word_indices, (num_lines, LINE_LENGTH), p=vocab_ps)
        print("Done!")
        np.savetxt(output_filename, vocab_samples, delimiter = " ", fmt = "%d")

if __name__ == "__main__":
    main()

