from random import shuffle

import torch
import pandas as pd
from torch.utils.data import IterableDataset


class CustomDataset(IterableDataset):
    """Custom dataset that reads one bitext chunk at a time.

    Parameters
    ----------
    bitext_file : str or list of str
        Path to the bitext file (*.csv) with two columns "source" and "target".
    src_vocab : str
        A dictionary obtained from the script
        `scripts/build_truncated_vocab.py` containing two keys: "tok2idx" and
        "idx2tok".
    tgt_vocab : str
        A dictionary obtained from the script
        `scripts/build_truncated_vocab.py` containing two keys: "tok2idx" and
        "idx2tok".
    num_samples : int
        Number of samples of this particular bitext chunk.
    shuffle : bool
        Whether to shuffle the dataset.
    """
    def __init__(self, bitext_file, src_vocab, tgt_vocab, num_samples=None,
                 shuffle=True):
        super(CustomDataset, self).__init__()
        self.bitext_file = bitext_file
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.shuffle = shuffle

        if num_samples is None:
            self.bitext = pd.read_csv(bitext_file)
            self.num_samples = len(self.bitext)
            self.bitext = None  # free memory
        else:
            self.bitext = None  # lazy init
            self.num_samples = num_samples
        # For random indexing
        self.idxs = list(range(self.num_samples))
        self.curr_idx = 0

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        if self.bitext is None:
            self.bitext = pd.read_csv(self.bitext_file)
        # Intialize
        if self.shuffle:
            shuffle(self.idxs)
        self.curr_idx = 0

        return self

    def __next__(self):
        # Stop if needed
        if self.curr_idx >= self.num_samples:
            self.bitext = None  # free memory
            raise StopIteration

        idx = self.idxs[self.curr_idx]
        src_text, tgt_text = self.bitext.loc[idx, ["source", "target"]]
        # Tokenize and convert to index
        src_tok2idx = self.src_vocab["tok2idx"]
        src_sos = src_tok2idx["<sos>"]
        src_eos = src_tok2idx["<eos>"]

        tgt_tok2idx = self.tgt_vocab["tok2idx"]
        tgt_sos = tgt_tok2idx["<sos>"]
        tgt_eos = tgt_tok2idx["<eos>"]

        src_toks = [src_tok2idx.get(tok, src_tok2idx["<unk>"])
                    for tok in src_text.split()]
        src_toks = [src_sos] + src_toks + [src_eos]

        tgt_toks = [tgt_tok2idx.get(tok, tgt_tok2idx["<unk>"])
                    for tok in tgt_text.split()]
        tgt_toks = [tgt_sos] + tgt_toks + [tgt_eos]

        # Update index
        self.curr_idx += 1
        return torch.tensor(src_toks), torch.tensor(tgt_toks)
