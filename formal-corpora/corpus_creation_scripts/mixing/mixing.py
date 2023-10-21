import argparse
from collections import deque
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
    parser.add_argument("--mix-prob", type=float, default = None)
    parser.add_argument("--open-prob", type=float, default = 0.49)
    parser.add_argument("--vocab-size", type=int, default = 50_000)
    parser.add_argument("--vocab-distribution", type=str, default = None)
    parser.add_argument("--deplength-distribution", type=str, default = None)
    parser.add_argument("--paired", action="store_true", default = False)
    args = parser.parse_args()
    print(args)
    output_dir = f"../../data/mixed-parens{args.mix_prob}"
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

    deplengths = json.load(open(os.path.join("dependency_lengths", f"deplengths_{args.deplength_distribution}.json"), 'r'))
    deplength_ps = np.array(list(deplengths.values())) / sum(deplengths.values())

    for split in ["valid", "test", "train"]:
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
        open_samples = np.random.choice([0, 1], (num_lines, LINE_LENGTH), p=[1 - args.open_prob, args.open_prob])
        deplength_keys = [int(key) for key in deplengths.keys()]
        deplength_samples = np.random.choice(deplength_keys, (num_lines, LINE_LENGTH), p=deplength_ps)
        mix_samples = np.random.choice([0, 1], (num_lines, LINE_LENGTH), p=[1 - args.mix_prob, args.mix_prob])
        print("Done!")
        for line_i in tqdm(range(num_lines), desc="[Generating samples]"):
            stack = deque()
            for word_i in range(LINE_LENGTH):
                if output[line_i, word_i] >= 0:
                    continue
                if mix_samples[line_i, word_i]:
                    chosen_word = vocab_samples[line_i, word_i]
                    output[line_i, word_i] = chosen_word
                    deplength = deplength_samples[line_i, word_i]
                    closing_index = word_i + deplength
                    if closing_index >= LINE_LENGTH:
                        continue
                    if output[line_i, closing_index] < 0:
                        output[line_i, closing_index] = match_index[chosen_word]
                    else:
                        # Look around the original sampled index to find the closest open
                        # spot.
                        displacement = 1
                        found_spot = False
                        while not found_spot and \
                            displacement < MAX_CLOSING_DISPLACEMENT and \
                            (closing_index - displacement >= 0 or closing_index + displacement < LINE_LENGTH):
                            if closing_index - displacement >= 0 and output[line_i, closing_index - displacement] < 0:
                                output[line_i, closing_index - displacement] = match_index[chosen_word]
                                found_spot = True
                            elif closing_index + displacement < LINE_LENGTH and output[line_i, closing_index + displacement] < 0:
                                output[line_i, closing_index + displacement] = match_index[chosen_word]
                                found_spot = True
                            displacement += 1
                else:
                    if open_samples[line_i, word_i] == 1 or len(stack) == 0:
                        output[line_i, word_i] = vocab_samples[line_i, word_i]
                        stack.append(output[line_i, word_i])
                    else:
                        output[line_i, word_i] = match_index[stack.pop()]
        np.savetxt(output_filename, output, delimiter = " ", fmt = "%d")

if __name__ == "__main__":
    main()
