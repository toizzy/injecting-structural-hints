import argparse
from collections import deque
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append('..')
from utils import get_distribution, LINE_LENGTH, CORPUS_LENGTHS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--open-prob", type=float, default = None)
    parser.add_argument("--vocab-size", type=int, default = 50_000)
    parser.add_argument("--vocab-distribution", type=str, default = None)
    parser.add_argument("--paired", action="store_true", default = False)
    args = parser.parse_args()
    print(args)
    output_dir = f"../../data/nested-parens{args.open_prob}"
    if args.vocab_size > 1_000:
        output_dir += f"_vocab{args.vocab_size // 1000}K"
    else:
        output_dir += f"_vocab{args.vocab_size}"
    output_dir += f"-{args.vocab_distribution}"
    if args.paired:
        output_dir += "_paired"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    print(f"Writing dataset to {output_dir}")
    if args.paired:
        word_indices, vocab_ps = get_distribution(args.vocab_distribution, args.vocab_size // 2)
        match_index = {word_i: args.vocab_size // 2 + word_i for word_i in word_indices}
    else:
        word_indices, vocab_ps = get_distribution(args.vocab_distribution, args.vocab_size)
        match_index = {word_i: word_i for word_i in word_indices}
    import IPython; IPython.embed()

    for split in ["valid", "train", "test"]:
        output_filename = os.path.join(output_dir, f"{split}.txt")
        print(f"On {split} split")
        split_length = CORPUS_LENGTHS[split]
        num_lines = split_length // LINE_LENGTH
        print(f"Going to make {num_lines} lines, for a corpus of {split_length} tokens")
        split_length = num_lines * LINE_LENGTH
        output = -1 * np.ones((num_lines, LINE_LENGTH)
        # Sampling all of the words in one go greatly increases speed. 
        # It's just flipping all coins in advance, does not actually change the process.
        print("Frontloading all random samples, which might take a bit")
        vocab_samples = np.random.choice(word_indices, (num_lines, LINE_LENGTH), p=vocab_ps)
        open_samples = np.random.choice([0, 1], (num_lines, LINE_LENGTH), p=[1 - args.open_prob, args.open_prob])
        print("Done!")
        for line_i in tqdm(range(num_lines), desc="[Generating samples]"):
            stack = deque()
            for word_i in range(LINE_LENGTH):
                if open_samples[line_i, word_i] == 1 or len(stack) == 0:
                    output[line_i, word_i] = vocab_samples[line_i, word_i]
                    stack.append(output[line_i, word_i])
                else:
                    output[line_i, word_i] = match_index[stack.pop()]
        np.savetxt(output_filename, output, delimiter = " ", fmt = "%d")

if __name__ == "__main__":
    main()
