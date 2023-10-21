import argparse
from collections import Counter
import json
import numpy as np
import os
import pickle
from tqdm import tqdm

import sys
sys.path.append('..')
from utils import get_distribution, LINE_LENGTH, CORPUS_LENGTHS

MAX_CLOSING_DISPLACEMENT = 3

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deplength-distribution", type=str, default = None)
    parser.add_argument("--vocab-size", type=int, default = 50_000)
    parser.add_argument("--vocab-distribution", type=str, default = None)
    parser.add_argument("--paired", action="store_true", default = False)
    args = parser.parse_args()
    output_dir = "../../data/flat-parens"
    if args.vocab_size > 1_000:
        output_dir += f"_vocab{args.vocab_size // 1000}K"
    elif args.vocab_size >  0:
        output_dir += f"_vocab{args.vocab_size}"
    output_dir += f"-{args.vocab_distribution}"
    if args.vocab_distribution == "matched":
        output_dir += f"-{args.vocab_distribution_file}"
    assert args.deplength_distribution
    output_dir += f"_deplength-{args.deplength_distribution}"
    if args.paired:
        output_dir += "_paired"
    os.mkdir(output_dir)
    print(f"Writing dataset to {output_dir}")
    word2idx = dict([(str(i), i) for i in range(args.vocab_size)])
    json.dump(word2idx, open(os.path.join(output_dir, f"vocab_limit{args.vocab_size}.json"), "w"))
    deplengths = json.load(open(os.path.join("dependency_lengths", f"deplengths_{args.deplength_distribution}.json"), 'r'))
    print(np.array(list(deplengths.values())))
    deplength_ps = np.array(list(deplengths.values())) / sum(deplengths.values())

    # If we're pairing open and close tokens, we pair the first half of the vocab size 
    # with the second half.
    if args.paired:
        word_indices, vocab_ps = get_distribution(args.vocab_distribution, args.vocab_size // 2)
        match_index = {word_i: args.vocab_size // 2 + word_i for word_i in word_indices}
    else:
        word_indices, vocab_ps = get_distribution(args.vocab_distribution, args.vocab_size)
        match_index = {word_i: word_i for word_i in word_indices}
    real_deplengths = Counter()

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
        deplength_keys = [int(key) for key in deplengths.keys()]
        deplength_samples = np.random.choice(deplength_keys, (num_lines, LINE_LENGTH), p=deplength_ps)
        print("Done!")
        for line_i in tqdm(range(num_lines), desc="[Generating samples]"):
            for word_i in range(LINE_LENGTH):
                # Check if this index is already taken by the closing
                # parenthesis of an earlier open.
                if output[line_i, word_i] >= 0:
                    continue
                chosen_word = vocab_samples[line_i, word_i]
                output[line_i, word_i] = chosen_word
                deplength = deplength_samples[line_i, word_i]
                closing_index = word_i + deplength
                if closing_index >= LINE_LENGTH:
                    continue
                if output[line_i, closing_index] < 0:
                    output[line_i, closing_index] = match_index[chosen_word]
                    real_deplengths[closing_index - word_i] += 1
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
                            real_deplengths[closing_index - displacement - word_i] += 1
                        elif closing_index + displacement < LINE_LENGTH and output[line_i, closing_index + displacement] < 0:
                            output[line_i, closing_index + displacement] = match_index[chosen_word]
                            found_spot = True
                            real_deplengths[closing_index + displacement - word_i] += 1
                        displacement += 1
        np.savetxt(output_filename, output, delimiter = " ", fmt = "%d")
    json.dump(deplengths, open(os.path.join(output_dir, "deplenghts.json"), "w"))

if __name__ == "__main__":
    main()

