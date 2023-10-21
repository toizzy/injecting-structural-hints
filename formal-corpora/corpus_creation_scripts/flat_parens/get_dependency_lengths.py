"""
Get the dependency lengths of a corpus. There are two options:
1) Use a UD file to get a dependency lengths distribution
2) Simulate nesting parentheses (depth-limiting optional) to see how dependency
"""
import argparse
from collections import deque, Counter
import conllu
import json
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("--ud-file", type=str, default=None)
parser.add_argument("--name", type=str, default=None)
parser.add_argument("--nesting-depth", type=int, default=-1)
args = parser.parse_args()

def main(args):
    nickname = ""
    if args.ud_file is not None:
        assert args.name is not None, "Please provide a nickname for this ud distribution using the --name arg"
        deplengths = get_dependency_lengths_ud(args.ud_file)
        nickname = args.name
    else:
        deplengths = get_dependency_lengths_nesting(args.nesting_depth)
        if args.name is not None:
            nickname = args.name
        else:
            if args.nesting_depth <= 0:
                nickname = "nesting-nolimit"
            else:
                nickname = f"nesting-limit{args.nesting_depth}"

    outfile = os.path.join("dependency_lengths", f"deplengths_{nickname}.json")
    json.dump(deplengths, open(outfile, "w"))

def get_dependency_lengths_nesting(depth):
    lengths = Counter()
    num_lines = 5_000
    line_length = 512
    # Flipping all coins in advance for speed
    open_decision = np.random.choice([0, 1], (num_lines, line_length))
    for line_i in range(num_lines):
        open_position_deque = deque()
        for word_i in range(line_length):
            if (open_decision[line_i, word_i] or len(open_position_deque) == 0) and \
               (depth <= 0 or len(open_position_deque) < depth):
               open_position_deque.append(word_i)
            else:
                last_open_position = open_position_deque.pop()
                lengths[word_i - last_open_position] += 1
    return lengths

    
def get_dependency_lengths_ud(ud_filename):
    with open(ud_filename, 'r') as f:
        conll_string = f.read()
    sentencelist = conllu.parse(conll_string)
    print(f"Parsed ud file at {ud_filename}!")
    dep_lengths = [abs(token['head'] -  token['id']) for sentence in sentencelist for token in sentence if token['head'] is not None]
    counter = Counter()
    for dep in dep_lengths:
        counter[dep] += 1
    return counter

main(args)
