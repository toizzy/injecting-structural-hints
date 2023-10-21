import argparse
from collections import Counter
import json
import numpy as np
import os
from pathlib import Path
import pickle
from tqdm import tqdm

LINE_LENGTH = 512
# Corpus length is in tokens, number of lines computed in the code
CORPUS_LENGTHS = {"train": 1_000_000_000, "test": 1_000_000, "valid": 500_000}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--len-repeating", type=int, default = 10)
    parser.add_argument("--vocab-size", type=int, default = 500)
    parser.add_argument("--vocab-distribution", type=str, default = "zipf")
    parser.add_argument("--paired", action="store_true", default = False)
    args = parser.parse_args()
    output_dir = f"../../data/simple_repetition{args.len_repeating}"
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
    json.dump(word2idx, open(output_dir / f"vocab_limit{args.vocab_size}.json", "w"))
    if args.paired:
        word_indices, vocab_ps = get_distribution(args.vocab_distribution, args.vocab_size // 2)
        match_index = {word_i: args.vocab_size // 2 + word_i for word_i in word_indices}
    else:
        word_indices, vocab_ps = get_distribution(args.vocab_distribution, args.vocab_size)
        match_index = {word_i: word_i for word_i in word_indices}
    for split in ["valid", "train", "test"]:
        output_filename = output_dir / f"{split}.txt"
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
        print("Done!")
        for line_i in tqdm(range(num_lines), desc="[Generating samples]"):
            for word_i in range(0, LINE_LENGTH, args.len_repeating * 2):
                segment_length = min(LINE_LENGTH - word_i, args.len_repeating * 2)
                segment = np.zeros(segment_length)
                for segment_i in range(min(segment_length, args.len_repeating)):
                    segment[segment_i] = vocab_samples[line_i, word_i + segment_i]
                for segment_i in range(args.len_repeating, min(segment_length, args.len_repeating * 2)):
                    segment[segment_i] = match_index[vocab_samples[line_i, word_i + segment_i - args.len_repeating]] 
                output[line_i, word_i : word_i + segment_length] = segment
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
    #if vocab_distribution:
    #    vocab = pickle.load(open(vocab_distribution, "rb"))
    #    most_common = vocab.most_common(vocab_size - 1)
    #    frequencies = [count for word, count in most_common]
    #    # Make an unk token, that has the probability of all other words
    #    unk_frequency = sum(vocab.values()) - sum(frequencies)
    #    frequencies.append(unk_frequency)
    #    frequencies = np.array(frequencies)
    #    np.random.shuffle(frequencies)
    #    ps = frequencies / sum(frequencies)
    #    word_indices = np.arange(len(ps))
    #    return word_indices, ps
if __name__ == "__main__":
    main()
