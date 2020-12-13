DESCRIPTION = """
Given the full vocab obtained from `build_vocab.py`, build the truncated vocab
into a json file with tok2idx and idx2tok fields.
"""

import sys
import json
import argparse

import pandas as pd


SPECIAL_TOKENS = ["<pad>", "<unk>", "<sos>", "<eos>"]


def main(args):
    args = vars(args)
    # Read full vocab
    full_vocab = pd.read_csv(args["full_vocab_path"], na_filter=False)
    # Process
    tokens = SPECIAL_TOKENS + full_vocab["word"].iloc[:args["vocab_size"]].to_list()
    tok2idx = dict(zip(tokens, range(len(tokens))))
    idx2tok = dict(zip(range(len(tokens)), tokens))
    # Save truncated vocab
    with open(args["save_path"], "w") as fout:
        json.dump({"tok2idx": tok2idx, "idx2tok": idx2tok}, fout)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=DESCRIPTION)
    parser.add_argument('-f', '--full-vocab-path', type=str, required=True,
        help='Path to the full vocab obtained from `build_vocab.py`.')
    parser.add_argument('-s', '--save-path', type=str, required=True,
        help='Path to save the truncated vocab (*.json).')
    parser.add_argument('--vocab-size', type=int, required=True,
        help='Vocab size (not counting special tokens).')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))