import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-constructions", type=str, default = None)
    parser.add_argument("--construction-length", type=str, default = None)
    parser.add_argument("--vocab-size", type=int, default = 50_000)
    parser.add_argument("--vocab-distribution", type=str, default = None)
    args = parser.parse_args()
    outfile = Path("construction_banks")
