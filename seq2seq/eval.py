"""
References
1. PyTorch tutorial: Language translation with torchtext.
   https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html
"""


import os
import math

import torch
import sacrebleu
import sacremoses
from loguru import logger


def evaluate(model, dataloader, criterion, device, prefix="", testing=False,
             search=None, save_to_file=None, print_samples=True, nbest=1,
             nprint=None, tgt_lang="fr", unescape=False):
    """Evaluate the trained model.

    The BLEU part was adapted from:
        https://github.com/pytorch/fairseq/blob/36c63c826d2292c9df56065b5816c02eefc87713/fairseq/tasks/translation.py#L414

    Parameters
    ----------
    model : nn.Module
        Initialized model.
    dataloader : torch.utils.data.DataLoader
        Test data loader.
    criterion : nn.Module
        Loss (objective) function.
    device : torch.device
        Device in which computation is performed.
    prefix : str
        Prefix for logging. (default: "")
    testing : bool
        If True, only run for 10 iterations. Useful for debugging and finding
        batch sizes, etc. (default: False)
    search : seq2seq.generator.SequenceGenerator
        Initialized `SequenceGenerator` object. (default: None)
    print_samples : bool
        If True, print `nprint` first elements of the first batch of results.
    save_to_file : str
        Path to save translated results (`fairseq` formatted) to.
        (default: None)
    nbest : int
        Number of top hypotheses to print out for each sample. Note that BLEU
        score will always calculated using the top-1 hypotheses regardless of
        `nbest`. (default: 1)
    nprint : int
        See `print_samples`.
    tgt_lang : str
        Target language. Used for detokenizer.
    unescape : False
        Set this to True if the training data was tokenized using `moses` with
        escaping, e.g., "'" gets turned into "&apos;". Defaults to False since
        the data in this project is preprocessed with `--no-escape`.
    """
    model.eval()
    epoch_loss = 0

    # Save BLEU results to a file
    if save_to_file is not None:
        save_to_dir = os.path.split(save_to_file)[0]
        os.makedirs(save_to_dir, exist_ok=True)

    if search is not None:
        logger.info("Starting beam search...")

        # Prepare
        src_vocab = dataloader.dataset.src_vocab
        src_pad = src_vocab["tok2idx"]["<pad>"]
        tgt_vocab = dataloader.dataset.tgt_vocab
        tgt_pad = tgt_vocab["tok2idx"]["<pad>"]

        # Tokenized strings
        src_sents, tgt_sents, tgt_hypo_sents = [], [], []
        scores, pos_scoress, idxs = [], [], []
        num_generated_tokens = 0

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if search is None:
                src, tgt = data
            else:
                src, tgt, sample_idxs = data

            src, tgt = src.to(device), tgt.to(device)

            # Forward
            output = model(src, tgt, 0)  # turn off teacher forcing

            # Compute loss
            output = output[1:].view(-1, output.shape[-1])
            tgt_ = tgt[1:].view(-1)
            loss = criterion(output, tgt_)

            # Update
            epoch_loss += loss.item()

            # Beam search and BLEU
            if search is not None:
                src = torch.transpose(src.long(), 0, 1)
                tgt = torch.transpose(tgt.long(), 0, 1)
                hypos = search.generate(src, prefix_tokens=None)

                # Avoid unexpected behavior with cuda
                src = src.cpu()
                tgt = tgt.cpu()

                for j, (sample_idx, src_toks, tgt_toks, tgt_hypos) in \
                        enumerate(zip(sample_idxs, src, tgt, hypos)):

                    # Remove padding from the source sentence
                    pad_mask = (src_toks == src_pad).int()
                    src_len = torch.argmin(pad_mask) - 1
                    src_toks = src_toks[1:src_len + 1]

                    # Remove padding from the target sentence
                    pad_mask = (tgt_toks == tgt_pad).int()
                    tgt_len = torch.argmin(pad_mask) - 1
                    tgt_toks = tgt_toks[1:tgt_len + 1]

                    # Reconstruct tokenized original source and target
                    # sentences
                    src_sent = " ".join([src_vocab["idx2tok"][str(tok_idx)]
                                         for tok_idx in src_toks.tolist()])
                    tgt_sent = " ".join([tgt_vocab["idx2tok"][str(tok_idx)]
                                         for tok_idx in tgt_toks.tolist()])

                    # Whether to print out results
                    do_print = (print_samples and i == 0)
                    if nprint is not None:
                        do_print = (do_print and j < nprint)

                    if do_print:
                        print(f"S-{sample_idx}\t{src_sent}")
                        print(f"T-{sample_idx}\t{tgt_sent}")

                    # Process top predictions
                    for k, tgt_hypo in enumerate(tgt_hypos[:nbest]):
                        # Remove padding from the target hypothesis. Note
                        # that <eos> has been removed from the hypotheses.
                        tgt_hypo_toks = tgt_hypo["tokens"].int().cpu()
                        pad_mask = (tgt_hypo_toks == tgt_pad).int()
                        tgt_hypo_len = torch.argmin(pad_mask)
                        tgt_hypo_toks = tgt_hypo_toks[:tgt_hypo_len]

                        # Reconstruct tokenized target hypothesis
                        tgt_hypo_sent = " ".join(
                            [tgt_vocab["idx2tok"][str(tok_idx)]
                             for tok_idx in tgt_hypo_toks.tolist()])

                        if do_print:
                            # Convert score to base 2
                            score = tgt_hypo["score"] / math.log(2)
                            # Tokenized hypothesis
                            print(f"H-{sample_idx}\t{score}\t"
                                  f"{tgt_hypo_sent}")
                            # Positional scores
                            pos_scores = tgt_hypo["positional_scores"]
                            pos_scores = map(
                                lambda x: f"{x:.4f}",
                                pos_scores.div_(math.log(2)).tolist())
                            pos_scores = " ".join(pos_scores)
                            print(f"P-{sample_idx}\t{pos_scores}")
                            print()

                        # Record only top hypotheses
                        if k == 0:
                            src_sents.append(src_sent)
                            tgt_sents.append(tgt_sent)
                            tgt_hypo_sents.append(tgt_hypo_sent)
                            scores.append(tgt_hypo["score"])
                            pos_scoress.append(
                                tgt_hypo["positional_scores"])
                            idxs.append(sample_idx)
                            num_generated_tokens += tgt_hypo_len

            # Break when reaching 10 iterations when testing
            if testing and i == 9:
                break

        if search is not None:
            logger.info(
                prefix
                + "NOTE: hypothesis and token scores are output in base 2")

            if save_to_file is None:
                logger.info(
                    prefix + f"Translated {len(tgt_sents)} sentences "
                    f"({num_generated_tokens} tokens)")
            else:
                logger.info(
                    prefix + f"Translated {len(tgt_sents)} sentences "
                    f"({num_generated_tokens} tokens) into "
                    f"{save_to_file}")
                # Write results into the output file
                with open(save_to_file, "w") as fout:
                    for (sample_idx, src_sent, tgt_sent, tgt_hypo_sent,
                         score, pos_scores) in \
                            zip(idxs, src_sents, tgt_sents,
                                tgt_hypo_sents, scores, pos_scoress):
                        fout.write(f"S-{sample_idx}\t{src_sent}\n")
                        fout.write(f"T-{sample_idx}\t{tgt_sent}\n")
                        fout.write(f"H-{sample_idx}\t{score}\t"
                                   f"{tgt_hypo_sent}\n")
                        fout.write(f"P-{sample_idx}\t{pos_scores}\n")

            # Calculate BLEU score
            tok_bleu = calc_tokenized_bleu(tgt_hypo_sents, [src_sents])
            detok_bleu = calc_detokenized_bleu(tgt_hypo_sents, [src_sents])

            logger.info(prefix + f"Tokenized BLEU: {tok_bleu.format()}")
            logger.info(
                prefix + f"Detokenized BLEU: {detok_bleu.format()}")

    model.train()
    return epoch_loss / (i + 1)


def calc_tokenized_bleu(hyp, refs):
    """Calculate tokenized BLEU using raw (tokenized) texts.

    References:
        https://github.com/pytorch/fairseq/blob/409032596bd80240f7fbc833b5d37485dee85b0e/fairseq/tasks/translation.py#L414
        https://github.com/pytorch/fairseq/blob/409032596bd80240f7fbc833b5d37485dee85b0e/fairseq_cli/score.py#L79

    Parameters
    ----------
    hyp : list of str
        A list containing hypotheses for each source sentence.
    refs : list of list of str
        A list of lists of candidate reference translations.

    Returns:
    sacrebleu.metrics.bleu.BLEUScore
        Tokenized BLEU score.
    """
    # Check for validity
    for ref in refs:
        assert len(ref) == len(hyp), ("Number of sentences in hypothesis and "
                                      f"reference does not match: {len(hyp)} "
                                      f"and {len(ref)}")

    # Make sure unknown words are escaped
    hyp = [hyp_sent.replace("<unk>", "<unk_hyp>") for hyp_sent in hyp]

    return sacrebleu.corpus_bleu(hyp, refs, tokenize="none")


def calc_detokenized_bleu(hyp, refs, tgt_lang="fr", unescape=False):
    """Calculate detokenized BLEU using raw (tokenized) texts. The input
    tokenized texts will be detokenized using `sacremoses`.

    References:
        https://github.com/pytorch/fairseq/blob/409032596bd80240f7fbc833b5d37485dee85b0e/fairseq/tasks/translation.py#L414
        https://github.com/pytorch/fairseq/blob/409032596bd80240f7fbc833b5d37485dee85b0e/fairseq_cli/score.py#L79

    Parameters
    ----------
    hyp : list of str
        A list containing hypotheses for each source sentence.
    refs : list of list of str
        A list of lists of candidate reference translations.
    tgt_lang : str
        Target language. Used for detokenizer.
    unescape : False
        Set this to True if the training data was tokenized using `moses` with
        escaping, e.g., "'" gets turned into "&apos;". Defaults to False since
        the data in this project is preprocessed with `--no-escape`.

    Returns:
    sacrebleu.metrics.bleu.BLEUScore
        Detokenized BLEU score.
    """
    # Check for validity
    for ref in refs:
        assert len(ref) == len(hyp), ("Number of sentences in hypothesis and "
                                      f"reference does not match: {len(hyp)} "
                                      f"and {len(ref)}")

    # Make sure unknown words are escaped
    hyp = [hyp_sent.replace("<unk>", "<unk_hyp>") for hyp_sent in hyp]

    # Detoknize
    detokenizer = sacremoses.MosesDetokenizer(lang=tgt_lang)
    hyp = [detokenizer.detokenize(hyp_sent.split()) for hyp_sent in hyp]
    refs_detok = []
    for ref in refs:
        ref = [detokenizer.detokenize(ref_sent.split(), unescape=unescape)
               for ref_sent in ref]
        refs_detok.append(ref)

    return sacrebleu.corpus_bleu(hyp, refs_detok, tokenize="none")
