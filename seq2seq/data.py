from random import choice

from loguru import logger
import torch
import pandas as pd
from torch.utils.data import Dataset


class CustomDataset(Dataset):
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
    def __init__(self, bitext_file, src_vocab, tgt_vocab, num_samples=None):
        super(CustomDataset, self).__init__()
        self.bitext_file = bitext_file
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        if num_samples is None:
            self.bitext = pd.read_csv(bitext_file, na_filter=False)
            self.num_samples = len(self.bitext)
            self.bitext = None  # free memory
        else:
            self.bitext = None  # lazy init
            self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if self.bitext is None:
            self.bitext = pd.read_csv(self.bitext_file, na_filter=False)
        try:
            return self.getitem(index)
        except Exception as e:  # return a random index
            logger.info(f"Error occured at index {index}: {e}")
            idx = choice(range(self.num_samples))
            return self.getitem(idx)

    def getitem(self, index):
        """Get item at a particular index"""
        src_text, tgt_text = self.bitext.loc[index, ["source", "target"]]
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

        return torch.tensor(src_toks), torch.tensor(tgt_toks)

    def clear(self):
        """Clear memory"""
        self.bitext = None


def collate_fn(batch, pad_idx=0):
    """Collate function to be passed to the PyTorch dataloader.

    Parameters
    ----------
    batch : list
        (uncollated) Batch containing `batch_size` pairs of sentences.
    pad_idx : int
        Index of the padding token "<pad>".

    Returns
    -------
    new_batch : [torch.Tensor, torch.Tensor]
        Collated batches (2 tensors), each of which is of shape
        (max_len, batch_size).

    """
    src_sents = [pair[0] for pair in batch]
    tgt_sents = [pair[1] for pair in batch]

    new_batch = []
    for sents in [src_sents, tgt_sents]:
        max_len = max(len(sent) for sent in sents)
        new_sents = torch.full(
            size=(max_len, len(batch)), fill_value=pad_idx, dtype=torch.int64)
        for i, sent in enumerate(sents):
            new_sents[:len(sent), i] = sent
        new_batch.append(new_sents)
    return new_batch
