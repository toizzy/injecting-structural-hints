import argparse
from collections import Counter
import json
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append('..')
from utils import get_distribution, LINE_LENGTH, CORPUS_LENGTHS

MAX_CLOSING_DISPLACEMENT = 3

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deplength", type=int, default = 10)
    parser.add_argument("--match-probability", type=float, default=0.1)
    parser.add_argument("--vocab-size", type=int, default = 50_000)
    parser.add_argument("--vocab-distribution", type=str, default = None)
    parser.add_argument("--paired", type=bool, default = False)
    args = parser.parse_args()
    output_dir = f"../../data/sparse{args.match_probability}-constant{args.deplength}"
    if args.vocab_size > 1_000:
        output_dir += f"_vocab{args.vocab_size // 1000}K"
    else:
        output_dir += f"_vocab{args.vocab_size}"
    output_dir += f"-{args.vocab_distribution}"
    if args.paired:
        output_dir += "_paired"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    word2idx = dict([(str(i), i) for i in range(args.vocab_size)])
    json.dump(word2idx, open(os.path.join(output_dir, f"vocab_limit{args.vocab_size}.json"), "w"))

    word_indices, vocab_ps = get_distribution(args.vocab_distribution, args.vocab_size)
    if args.paired:
        word_indices, vocab_ps = get_distribution(args.vocab_distribution, args.vocab_size // 2)
        match_index = {word_i: args.vocab_size // 2 + word_i for word_i in word_indices}
    else:
        word_indices, vocab_ps = get_distribution(args.vocab_distribution, args.vocab_size)
        match_index = {word_i: word_i for word_i in word_indices}
    
    for split in ["valid", "train", "test"]:
        output_filename = os.path.join(output_dir, f"{split}.txt")
        print(f"On {split} split")
        split_length = CORPUS_LENGTHS[split]
        num_lines = split_length // LINE_LENGTH
        print(f"Going to make {num_lines} lines, for a corpus of {split_length} tokens")
        split_length = num_lines * LINE_LENGTH
        output = -1 * np.ones((num_lines, LINE_LENGTH))
        # Sampling all of the words in one go greatly increases speed. 
        # Just flipping all coins in advance, does not change process.
        print("Frontloading all random samples, which might take a bit")
        vocab_samples = np.random.choice(word_indices, (num_lines, LINE_LENGTH), p=vocab_ps)
        match_samples = np.random.choice([0, 1], (num_lines, LINE_LENGTH), p=[1 - args.match_probability, args.match_probability])
        print("Done!")
        for line_i in tqdm(range(num_lines), desc="[Generating samples]"):
            for word_i in range(LINE_LENGTH):
                # Check if this index is already taken by the closing
                # parenthesis of an earlier open.
                if output[line_i, word_i] >= 0:
                    continue
                chosen_word = vocab_samples[line_i, word_i]
                output[line_i, word_i] = chosen_word
                if match_samples[line_i, word_i] == 1:
                    if word_i < LINE_LENGTH - args.deplength:
                        output[line_i, word_i + args.deplength] = match_index[chosen_word]
        np.savetxt(output_filename, output, delimiter = " ", fmt = "%d")

if __name__ == "__main__":
    main()
