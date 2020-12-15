import sys
import math
import argparse
from collections import Counter

import pandas as pd


DESCRIPTION = """
Build full vocabulary from list of csv files consisting of "source" and
"target" columns. Write into a csv file.
"""


def main(args):
    args = vars(args)
    show_progress = args["show_progress"]
    lowercase = args["lowercase"]

    # Progress bar
    if show_progress:
        from tqdm import tqdm
    else:
        tqdm = lambda x: x

    src_counter = {}
    tgt_counter = {}

    # Start reading
    for filepath in args["file_paths"]:
        print(f"Reading {filepath}")
        bitext = pd.read_csv(filepath)
        bitext.dropna(inplace=True)  # some may have NaN values
        if lowercase:
            bitext["source"] = bitext["source"].str.lower()
            bitext["target"] = bitext["target"].str.lower()
        # Loop over each batch
        batch_size = args["batch_size"]
        num_batches = math.ceil(len(bitext) / batch_size)
        for batch in tqdm(range(num_batches)):
            start = batch * batch_size
            end = min(start + batch_size, len(bitext))
            # Start counting
            src_batch_text = bitext["source"].iloc[start:end]
            src_batch_text = " ".join(src_batch_text)
            src_counter.update(Counter(src_batch_text.split()))

            tgt_batch_text = bitext["target"].iloc[start:end]
            tgt_batch_text = " ".join(tgt_batch_text)
            tgt_counter.update(Counter(tgt_batch_text.split()))
        # Free memory
        del bitext

    # Save counter to a file
    src_df = pd.DataFrame.from_dict(
        src_counter, orient="index", columns=["count"]).reset_index()
    src_df = src_df.rename({"index": "word"}, axis=1).sort_values(
        "count", ascending=False)
    src_df.to_csv(args["src_output"], index=False)

    tgt_df = pd.DataFrame.from_dict(
        tgt_counter, orient="index", columns=["count"]).reset_index()
    tgt_df = tgt_df.rename({"index": "word"}, axis=1).sort_values(
        "count", ascending=False)
    tgt_df.to_csv(args["tgt_output"], index=False)
    print(f"Done saving vocabulary to {args['src_output']} and "
          f"{args['tgt_output']}")


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=DESCRIPTION)
    parser.add_argument(
        '-f', '--file-paths', type=str, nargs='+', required=True,
        help='Path to all the csv files separate by space.')
    parser.add_argument(
        '-s', '--src-output', type=str, required=True,
        help='Path to save the source language vocabulary.')
    parser.add_argument(
        '-r', '--tgt-output', type=str, required=True,
        help='Path to save the target language vocabulary.')
    parser.add_argument(
        '--lowercase', action="store_true",
        help='Whether to lowercase all texts.')
    parser.add_argument(
        '--show-progress', action="store_true",
        help='Whether to show progress bar.')
    parser.add_argument(
        '--batch-size', type=int, default=30000,
        help='Process in batches.')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
