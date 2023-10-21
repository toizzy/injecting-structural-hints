import argparse
from collections import Counter
import json
import numpy as np
import os
from pathlib import Path
import pickle
import random
from tqdm import tqdm

LINE_LENGTH = 512
# Corpus length is in tokens, number of lines computed in the code
CORPUS_LENGTHS = {"train": 1_000_000_000, "test": 1_000_000, "valid": 500_000}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--len-repeating", type=int, default = 10)
    parser.add_argument("--mod", type=int, default = 10)
    parser.add_argument("--vocab-size", type=int, default = 500)
    parser.add_argument("--vocab-distribution", type=str, default = "zipf-simple")
    args = parser.parse_args()
    output_dir = f"../../data/mod{args.mod}_repetition{args.len_repeating}"
    if args.vocab_size != 50_000:
        # Only mark vocab size on filename if it's not the default, to reduce filename cluttering
        output_dir += f"_vocabsize-{args.vocab_size}"
    if args.vocab_distribution is not None:
        output_dir += f"_vocab-{args.vocab_distribution}"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    word2idx = dict([(str(i), i) for i in range(args.vocab_size)])
    json.dump(word2idx, open(output_dir / f"vocab_limit{args.vocab_size}.json", "w"))
    word_indices, vocab_ps = get_distribution(args.vocab_distribution, args.vocab_size)
    num_quotients = (len(word_indices) // args.mod) + 1
    remainders_dist, quotients_dists = get_remainder_quotient_distribution(word_indices, vocab_ps, args.mod)
    for split in ["valid", "train", "test"]:
        output_filename = output_dir / f"{split}.txt"
        print(f"On {split} split")
        split_length = CORPUS_LENGTHS[split]
        num_lines = split_length // LINE_LENGTH
        print(f"Going to make {num_lines} lines, for a corpus of {split_length} tokens")
        split_length = num_lines * LINE_LENGTH
        output = -1 * np.ones((num_lines, LINE_LENGTH))
        # Sampling all of the words in one go greatly increases speed. 
        # It's just flipping all coins in advance, does not change process.
        print("Frontloading all random samples, which might take a bit")
        # Since we're doing mods, we need to sample the remainders and the quotients separately.
        # So, say we're mod 10, and the remainder is 1. To know what number we will actually
        # put, we look at the quotient. Say it's 4, we put 41
        remainder_samples = np.random.choice(range(args.mod), (num_lines, LINE_LENGTH), p=remainders_dist)
        print("Done!")
        for line_i in tqdm(range(num_lines), desc="[Generating samples]"):
            for word_i in range(0, LINE_LENGTH, args.len_repeating * 2):
                segment_length = min(LINE_LENGTH - word_i, args.len_repeating * 2)
                segment = np.zeros(segment_length)
                for segment_i in range(min(segment_length, args.len_repeating)):
                    remainder = remainder_samples[line_i, word_i + segment_i]
                    quotient = random.choices(range(num_quotients), weights=quotients_dists[remainder], k=1)[0]
                    segment[segment_i] = quotient*args.mod + remainder
                for segment_i in range(args.len_repeating, min(segment_length, args.len_repeating * 2)):
                    remainder = remainder_samples[line_i, word_i + segment_i - args.len_repeating]
                    quotient = random.choices(range(num_quotients), weights=quotients_dists[remainder], k=1)[0]
                    segment[segment_i] = quotient*args.mod + remainder
                output[line_i,word_i : word_i + segment_length] = segment
        np.savetxt(output_filename, output, delimiter = " ", fmt = "%d")


def get_distribution(vocab_distribution, vocab_size):
    assert vocab_size > 0, "Need a --vocab-size > 0, even if --vocab-distribution is provided"
    if vocab_distribution == "zipf-simple":
        word_indices = np.arange(vocab_size)
        np.random.shuffle(word_indices)
        ps = [1 / (r + 2.7) for r in range(vocab_size)]
        ps = ps / np.sum(ps)
        print(word_indices)
        print(ps)
        return word_indices, ps
    else:
        word_indices = np.arange(vocab_size)
        ps = np.ones(vocab_size)
        uniform_probability = 1.0 / vocab_size
        ps = ps * uniform_probability
        return word_indices, ps
    if vocab_distribution == "dummy_not_implemented":
        vocab = pickle.load(open(vocab_distribution, "rb"))
        most_common = vocab.most_common(vocab_size - 1)
        frequencies = [count for word, count in most_common]
        # Make an unk token, that has the probability of all other words
        unk_frequency = sum(vocab.values()) - sum(frequencies)
        frequencies.append(unk_frequency)
        frequencies = np.array(frequencies)
        np.random.shuffle(frequencies)
        ps = frequencies / sum(frequencies)
        word_indices = np.arange(len(ps))
        return word_indices, ps

def get_remainder_quotient_distribution(word_indices, vocab_ps, mod):
    remainders_distribution = np.zeros(mod)
    num_quotients = (len(word_indices) // mod) + 1
    marg_quotient_distributions = \
        {remainder: np.zeros(num_quotients) for remainder in range(mod)}
    for word_idx, p in zip(word_indices, vocab_ps):
        remainder = word_idx % mod
        remainders_distribution[remainder] += p
        quotient = word_idx // mod
        marg_quotient_distributions[remainder][quotient] = p
    for remainder in range(mod):
        if sum(marg_quotient_distributions[remainder]) > 0:
            marg_quotient_distributions[remainder] = \
                marg_quotient_distributions[remainder] / sum(marg_quotient_distributions[remainder])
    return remainders_distribution, marg_quotient_distributions
    
def test():
    word_indices = range(20)
    ps = np.zeros(20)
    ps[1] = 0.25
    ps[11] = 0.25
    ps[8] = 0.25
    ps[16] = 0.25
    print(get_remainder_quotient_distribution(word_indices, ps, 10))


if __name__ == "__main__":
    main()
