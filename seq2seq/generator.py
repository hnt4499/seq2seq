import math

import torch
import torch.nn as nn
from loguru import logger


class BeamSearch:
    """Beam search.

    Adapted from
        https://github.com/pytorch/fairseq/blob/master/fairseq/search.py

    Parameters
    ----------
    tgt_vocab : dict
        Target language's Vocabulary.
    """
    def __init__(self, tgt_vocab):
        self.tgt_vocab = tgt_vocab

    def step(self, step, curr_lprobs, scores):
        """Take a single search step.

        Parameters
        ----------
        step : int
            Current timestep.
        curr_lprobs : torch.Tensor
            The model's log-probabilities over the vocabulary at the current
            step of shape (batch, input_beam_size, tgt_vocab_size).
        scores : torch.Tensor
            The historical model scores of each hypothesis up to the current
            step of shape (batch, input_beam_size, max_len).

        Returns
        -------
        curr_scores : torch.Tensor
            The scores of the chosen elements of shape (batch,
            output_beam_size). Note that output_beam_size can be larger than
            input_beam_size, e.g., we may return 2 * input_beam_size to account
            for EOS.
        curr_indices : torch.Tensor
            The indices of the chosen elements of shape (batch,
            output_beam_size).
        curr_beams : torch.Tensor
            The hypothesis indices of the chosen elements, in the range
            [0, input_beam_size). A tensor of shape (batch, output_beam_size).
        """
        batch, input_beam_size, tgt_vocab_size = curr_lprobs.size()

        if step == 0:
            # At the first step all hypotheses are equally likely, so use
            # only the first beam; of shape # (batch, 1, tgt_vocab_size)
            curr_lprobs = curr_lprobs[:, :1, :].contiguous()
        else:
            # Add curr_probs to the cumulative scores for each hypothesis
            curr_lprobs = curr_lprobs + scores[:, :, step - 1].unsqueeze(
                -1)  # (batch, input_beam_size, tgt_vocab_size)

        top_prediction = torch.topk(
            curr_lprobs.view(batch, -1),
            k=min(
                # Take the best 2 x beam_size predictions.
                input_beam_size * 2,
                input_beam_size * (tgt_vocab_size - 1)  # -1 to avoid <pad>
            ),
        )
        curr_scores = top_prediction[0]  # (batch, top_k)
        curr_indices = top_prediction[1]  # (batch, top_k)

        # Project back into relative indices and beams
        curr_beams = curr_indices // tgt_vocab_size  # (batch, top_k)
        curr_indices = curr_indices.fmod(tgt_vocab_size)  # (batch, top_k)

        return curr_scores, curr_indices, curr_beams


class SequenceGenerator(nn.Module):
    def __init__(self, model, src_vocab, tgt_vocab, beam_size=1, max_len_a=0,
                 max_len_b=200, min_len=1, normalize_scores=True,
                 len_penalty=1.0, unk_penalty=0.0, temperature=1.0):
        """Generates translations of a given source sentence.

        Args:
            model : nn.Module
                PyTorch model.
            src_vocab : dict
                Source language's vocabulary.
            tgt_vocab : dict
                Target language's vocabulary.
            beam_size : int
                Beam width. (default: 1)
            max_len_a/b : int
                Generate sequences of maximum length ax + b, where x is the
                source length (default: a: 0; b: 200; i.e., independent of
                source sentence lengths.)
            min_len : int
                The minimum length of the generated output (not including
                end-of-sentence). (default: 1)
            normalize_scores : bool
                Normalize scores by the length of the output (default: True)
            len_penalty : float
                Length penalty, where < 1.0 favors shorter, > 1.0 favors
                longer sentences. (default: 1.0)
            unk_penalty : float
                Unknown word penalty, where < 0 produces more unks, > 0
                produces fewer. (default: 0.0)
            temperature : float
                Temperature, where values > 1.0 produce more uniform samples
                and values < 1.0 produce sharper samples. (default: 1.0)
        """
        super().__init__()

        self.src_vocab = src_vocab
        self.src_pad = src_vocab["tok2idx"]["<pad>"]
        self.src_eos = src_vocab["tok2idx"]["<eos>"]

        self.tgt_vocab = tgt_vocab
        self.tgt_pad = tgt_vocab["tok2idx"]["<pad>"]
        self.tgt_unk = tgt_vocab["tok2idx"]["<unk>"]
        self.tgt_sos = tgt_vocab["tok2idx"]["<sos>"]
        self.tgt_eos = tgt_vocab["tok2idx"]["<eos>"]

        self.vocab_size = len(tgt_vocab["tok2idx"])
        self.beam_size = min(beam_size, self.vocab_size - 1)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len

        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.temperature = temperature

        self.unk_penalty = unk_penalty
        if unk_penalty < 0:
            logger.warn(f"You should probably never want to produce more "
                        f"unknown words with `unk_penalty={unk_penalty}`")

        self.search = BeamSearch(tgt_vocab)
        self.model = model
        self.model.eval()

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def generate(self, src_tokens, prefix_tokens=None):
        """Generate a batch of translations.

        Adapted from
            https://github.com/pytorch/fairseq/blob/c8a0659be5cdc15caa102d5bbf72b872567c4859/fairseq/sequence_generator.py#L178
        where we assume to have only one moodel (instead of an ensemble of
        models).

        Parameters
        ----------
        src_tokens : torch.LongTensor
            Source sentences of shape (batch, src_len). Source sentences to
            translate from (after after tokenizing, indexing and padding).
        prefix_tokens : torch.LongTensor
            Force decoder to begin with these tokens (possible of different
            lenghts) of shape (batch, *).

        Returns
        -------
        finalized : List[List[Dict[str, Tensor]]]
            A list of list of hypotheses, where `len(finalized) = batch_size`,
            and `len(finalized[i]) = beam_size`. The innermost dictionary
            contains 5 keys: `tokens` (translated tokens), `score` (score
            (probability) of the translated sentence), `attention`
            (soft-attention score between the source and target sentence), and
            `positional_scores`. Note that the order of the samples is
            preserved (e.g., we can do something like
            `zip(src_tokens, finalized)`), and the hypotheses for each sample
            have been sorted descendingly (higher score is better).
        """
        # Length of source texts being the number of characters except <eos>
        # and <pad>
        src_lengths = (
            (src_tokens.ne(self.src_eos) & src_tokens.ne(self.src_pad))
            .long().sum(dim=1)
        )  # (batch,)

        batch, src_len = src_tokens.shape
        beam_size = self.beam_size
        max_len = int(self.max_len_a * src_len + self.max_len_b)
        if max_len < self.min_len:
            raise RuntimeError(f"`max_len` ({max_len}) cannot be smaller than "
                               f"`min_len` ({self.min_len})")

        # Compute the encoder output for input sentence.
        # enc_outputs: (src_len, batch, num_directions * enc_hid_dim)
        # decoder_hidden: (batch, dec_hid_dim)
        enc_outputs, decoder_hidden = self.model.encoder(
            torch.transpose(src_tokens, 0, 1).long())

        # Reorder encoder outputs
        new_order = torch.arange(batch).view(-1, 1).repeat(
            1, beam_size).view(-1)  # (batch * beam_size)
        new_order = new_order.to(src_tokens.device).long()
        # where new_order == [0, 1, ..., batch, 0, 1, ..., batch, 0, 1, ...]

        # enc_outputs: (src_len, batch * beam_size, num_directions*enc_hid_dim)
        # decoder_hidden: (batch * beam_size, dec_hid_dim)
        enc_outputs, decoder_hidden = self.model.encoder.reorder_encoder_out(
            (enc_outputs, decoder_hidden), new_order)

        # Initialize buffers to hold tokens and accumulative scores
        scores = torch.zeros(batch * beam_size, max_len + 1).to(
            src_tokens.device).float()  # + 1 for <eos>
        tokens = (
            torch.zeros(batch * beam_size, max_len + 2)
            .to(src_tokens)
            .long()
            .fill_(self.tgt_pad)
        )  # + 2 for <sos> and <eos>
        tokens[:, 0] = self.tgt_sos

        # A list that indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then cands_to_ignore would mark 2 positions as being
        # ignored so that we only finalize the remaining 3 samples.
        cands_to_ignore = (
            torch.zeros(batch, beam_size).to(src_tokens).eq(-1)
        )  # forward and backward-compatible False mask

        # List of finalized/finished sentences
        finalized = [[] for i in range(batch)]
        finished = [False] * batch
        num_remaining_sent = batch  # number of sentences remaining

        # Number of candidate hypos per step (* 2 to account for <eos>)
        cand_size = beam_size * 2

        # Offset arrays for converting between different indexing schemes
        # batch_offsets: (batch, 1); cand_offsets: (cand_size,)
        batch_offsets = (torch.arange(batch) * beam_size).unsqueeze(1).type_as(
            tokens).to(src_tokens)
        cand_offsets = torch.arange(cand_size).to(src_tokens)

        reorder_state = None
        attn = None  # to record attention scores
        batch_idxs = None

        # One extra step for <eos>
        for step in range(max_len + 1):
            # Reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # Update beam indices to take into acc. removed sentences
                    corr = (
                        batch_idxs
                        - torch.arange(batch_idxs.numel()).type_as(batch_idxs))
                    reorder_state.view(-1, beam_size).add_(
                        corr.unsqueeze(-1) * beam_size)
                # We should have reordered incremental states of the decoder
                # here, but it turns out to be the last hidden state of the
                # encoder (`decoder_hidden` below). Hence, we need only reorder
                # encoder outputs
                enc_outputs, decoder_hidden = (
                    self.model.encoder.reorder_encoder_out(
                        (enc_outputs, decoder_hidden), reorder_state)
                )

            # lprobs: (batch * beam_size, tgt_vocab_size)
            # decoder_hidden: (batch * beam_size, dec_hid_dim)
            # attn_scores: (batch, src_len)
            lprobs, decoder_hidden, attn_scores = self.model.forward_decoder(
                tokens[:, step],  # (batch * beam_size,)
                decoder_hidden,  # (batch * beam_size, dec_hid_dim)
                # (seq_len, batch * beam_size, num_directions * enc_hid_dim)
                enc_outputs,
                self.temperature,
            )

            # lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

            lprobs[:, self.tgt_pad] = -math.inf  # never select pad
            lprobs[:, self.tgt_unk] -= self.unk_penalty  # apply unk penalty

            # Handle max length constraint
            if step >= max_len:
                lprobs[:, :self.tgt_eos] = -math.inf
                lprobs[:, self.tgt_eos + 1:] = -math.inf

            # Handle prefix tokens (possibly with different lengths)
            if (
                prefix_tokens is not None
                and step < prefix_tokens.size(1)
                and step < max_len
            ):
                lprobs, tokens, scores = self._prefix_tokens(
                    step, lprobs, scores, tokens, prefix_tokens, beam_size
                )
            elif step < self.min_len:
                # Minimum length constraint (does not apply if using
                # prefix_tokens)
                lprobs[:, self.tgt_eos] = -math.inf

            # Record attention scores
            if attn is None:
                attn = torch.empty(
                    batch * beam_size, attn_scores.size(1), max_len + 2
                ).to(scores)  # (batch * beam_size, src_len, max_len + 2)
            attn[:, :, step + 1].copy_(attn_scores)

            scores = scores.type_as(lprobs)
            eos_batch_idx = torch.empty(0).to(
                tokens
            )  # indices of hypothesis ending with eos (finished sentences)
            eos_scores = torch.empty(0).to(
                scores
            )  # scores of hypothesis ending with eos (finished sentences)

            # All are of shape (batch, cand_size)
            cand_scores, cand_indices, cand_beams = self.search.step(
                step,
                # (batch, beam_size, tgt_vocab_size)
                lprobs.view(batch, -1, self.vocab_size),
                # (batch, beam_size, max_len + 1)
                scores.view(batch, beam_size, -1),
            )

            # cand_batch_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, batch * beam_size),
            # and dimensions: (batch, cand_size)
            cand_batch_idx = cand_beams.add(batch_offsets)

            # Finalize hypotheses that end in <eos>
            eos_mask = (
                cand_indices.eq(self.tgt_eos) & cand_scores.ne(-math.inf)
            )  # (batch, cand_size)
            # Ignore hypotheses ending in <eos> but already finalized
            eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(
                eos_mask)

            # Only consider <eos> when it's among the top `beam_size` indices
            # Now we know what beam item(s) to finish. Note that this might
            # contain multiple hypotheses of the same source sentences
            eos_batch_idx = torch.masked_select(
                cand_batch_idx[:, :beam_size], mask=eos_mask[:, :beam_size]
            )  # (n,), where n is the number of un-finalized hypotheses ending
            # in <eos> among the top `beam_size` indices. `n` might be larger
            # than `beam_size` or `batch`.

            finalized_sents = []
            if eos_batch_idx.numel() > 0:
                eos_scores = torch.masked_select(
                    cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size]
                )  # (n,)

                finalized_sents = self.finalize_hypos(
                    step,  # current step
                    eos_batch_idx,  # (n,)
                    eos_scores,  # (n,)
                    tokens,  # (batch * beam_size, max_len + 2)
                    scores,  # (batch * beam_size, max_len + 1)
                    finalized,  # list of size `batch` of finalized hypotheses
                    finished,  # list of size `batch` of finished hypotheses
                    beam_size,  # beam width
                    attn,  # (batch * beam_size, src_len, max_len + 2)
                    src_lengths,  # lengths of source sentences; shape (batch,)
                    max_len,  # max translated length
                )  # list of size `m`, where m is the number of newly finalized
                # sentence(s).
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            if step >= max_len:
                break
            assert step < max_len

            # Remove finalized sentences (ones for which enough `beam_size`
            # finished hypotheses have been generated) from the batch.
            if len(finalized_sents) > 0:
                new_batch = batch - len(finalized_sents)

                # Construct batch_idxs which holds indices of batches to keep
                # for the next pass
                batch_mask = torch.ones(
                    batch, dtype=torch.bool, device=cand_indices.device
                )  # (batch,)
                batch_mask[finalized_sents] = False  # (batch,)
                batch_idxs = torch.arange(
                    batch, device=cand_indices.device
                ).masked_select(batch_mask)  # (new_batch,)

                # Update tensors for the new iteration
                eos_mask = eos_mask[batch_idxs]  # (new_batch, cand_size)
                cand_beams = cand_beams[batch_idxs]  # (new_batch, cand_size)
                batch_offsets.resize_(new_batch, 1)  # (new_batch, 1)
                cand_batch_idx = cand_beams.add(
                    batch_offsets)  # (new_batch, cand_size)
                cand_scores = cand_scores[batch_idxs]  # (new_batch, cand_size)
                cand_indices = cand_indices[
                    batch_idxs]  # (new_batch, cand_size)

                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]  # (new_batch, *)
                src_lengths = src_lengths[batch_idxs]  # (new_batch,)
                cands_to_ignore = cands_to_ignore[
                    batch_idxs]  # (new_batch, beam_size)

                scores = scores.view(batch, -1)[batch_idxs].view(
                    new_batch * beam_size,
                    -1)  # (new_batch * beam_size, max_len + 1)
                tokens = tokens.view(batch, -1)[batch_idxs].view(
                    new_batch * beam_size,
                    -1)  # (new_batch * beam_size, max_len + 2)

                attn = attn.view(batch, -1)[batch_idxs].view(
                    new_batch * beam_size, attn.size(1), -1
                )  # (new_batch, src_len, max_len + 2)
                # or (new_batch, src_len, tgt_len)

                # Update batch size
                batch = new_batch

            else:
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos.
            # Note that the cand_offsets is used to preserve the beam orders.
            eos_mask[:, :beam_size] = ~(
                (~cands_to_ignore) & (~eos_mask[:, :beam_size])
            )  # (new_batch, cand_size)
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[:eos_mask.size(1)],
            )  # (new_batch, cand_size)

            # Get the top beam_size active hypotheses, which are just the hypos
            # with the smallest values in active_mask. {active_hypos} indicates
            # which {beam_size} hypotheses from the list of {2 * beam_size}
            # candidates were selected.
            new_cands_to_ignore, active_hypos = torch.topk(
                active_mask, k=beam_size, dim=1, largest=False
            )  # (new_batch, beam_size) and (new_batch, beam_size)

            # Update cands_to_ignore to ignore any finalized hypos.
            cands_to_ignore = new_cands_to_ignore.ge(cand_size)[
                :, :beam_size]  # (new_batch, beam_size)
            # Make sure there is at least one active item for each sentence in
            # the batch.
            assert (~cands_to_ignore).any(dim=1).all()

            # Update cands_to_ignore to ignore any finalized hypos

            # {active_batch_idx} denotes which beam number is continued for
            # each new hypothesis (a beam can be selected more than once).
            active_batch_idx = torch.gather(
                cand_batch_idx, dim=1, index=active_hypos
            ).view(-1)  # (new_batch * beam_size)

            # Copy tokens and scores for active hypotheses

            # Set the tokens for each beam (can select the same row more than
            # once)
            tokens[:, :step + 1] = torch.index_select(
                tokens[:, :step + 1], dim=0, index=active_batch_idx
            )  # (new_batch * beam_size, max_len + 2)

            # Select the next token for each of them
            tokens.view(batch, beam_size, -1)[:, :, step + 1] = torch.gather(
                cand_indices, dim=1, index=active_hypos
            )  # (new_batch, beam_size, max_len + 2)
            if step > 0:
                scores[:, :step] = torch.index_select(
                    scores[:, :step], dim=0, index=active_batch_idx
                )  # (new_batch * beam_size, max_len + 1)
            scores.view(batch, beam_size, -1)[:, :, step] = torch.gather(
                cand_scores, dim=1, index=active_hypos
            )  # (new_batch, beam_size, max_len + 1)

            # Copy attention for active hypotheses
            attn[:, :, :step + 2] = torch.index_select(
                attn[:, :, :step + 2], dim=0, index=active_batch_idx
            )  # (new_batch, src_len, max_len + 2)
            # or (new_batch, src_len, tgt_len)

            # reorder incremental state in decoder
            reorder_state = active_batch_idx

        # Sort by score descending
        for sent_idx in range(len(finalized)):
            scores = torch.tensor(
                [float(elem["score"].item()) for elem in finalized[sent_idx]]
            )
            _, sorted_scores_indices = torch.sort(scores, descending=True)
            finalized[sent_idx] = [finalized[sent_idx][ssi]
                                   for ssi in sorted_scores_indices]
        return finalized

    def _prefix_tokens(self, step, lprobs, scores, tokens, prefix_tokens,
                       beam_size):
        """Handle prefix tokens by changing log-probabilities matrix at a
        particular timestep according to the given prefix tokens.

        Adapted from
            https://github.com/pytorch/fairseq/blob/c8a0659be5cdc15caa102d5bbf72b872567c4859/fairseq/sequence_generator.py#L542

        Parameters
        ----------
        step : int
            Current time step to handle.
        lprobs : torch.Tensor
            Normalized decoder output (log-probabilities) of shape
            (batch * beam_size, tgt_vocab_size).
        scores : torch.Tensor
            A tensor of shape (batch * beam_size, max_len + 1) storing
            accumulative beam scores.
        tokens : torch.Tensor
            A tensor of shape (batch * beam_size, max_len + 2) storing
            generated tokens (including
            <sos> tokens).
        prefix_tokens : torch.Tensor
            Force decoder to begin with these tokens (possible of different
            lengths)
            of shape (batch, n).
        beam_size : int
            Beam width.

        Returns
        -------
        tuple of (lprobs, tokens, scores) where the tensors are handled by
        changing log-probabilities matrix at a particular timestep according to
        the given prefix tokens.
        """
        prefix_toks = (
            prefix_tokens[:, step]  # prefix tokens at the current timestep
            .unsqueeze(-1)
            .repeat(1, beam_size)
            .view(-1)
        )  # (batch * beam_size,)

        # Get log-probabilities of the prefix tokens
        prefix_lprobs = lprobs.gather(
            -1, prefix_toks.unsqueeze(-1))  # (batch * beam_size,)
        prefix_mask = prefix_toks.ne(
            self.tgt_pad)  # (batch * beam_size,)

        # Set probs. of all beams that are currently not <pad> to -inf
        lprobs[prefix_mask] = torch.tensor(-math.inf).to(
            lprobs)  # (batch * beam_size, tgt_vocab_size)

        # On top of -inf (above step), for each sentence, fill
        # the token that is the prefix with its corresponding
        # probability (and leave all other tokens -inf)
        lprobs[prefix_mask] = lprobs[prefix_mask].scatter(
            dim=-1, index=prefix_toks[prefix_mask].unsqueeze(-1),
            src=prefix_lprobs[prefix_mask]
        )

        # If prefix includes <eos>, then we should make sure tokens and
        # scores are the same across all beams
        eos_mask = prefix_toks.eq(self.tgt_eos)  # (batch * beam_size,)
        if eos_mask.any():
            # Validate that the first beam matches the prefix
            first_beam = tokens[eos_mask].view(-1, beam_size, tokens.size(-1))[
                :, 0, 1:step + 1
            ]
            # Mask of shape (batch,) indicating whether a prefix sentence
            # includes <eos>
            eos_mask_batch_dim = eos_mask.view(-1, beam_size)[:, 0]  # (batch,)
            target_prefix = prefix_tokens[eos_mask_batch_dim][:, :step]
            assert (first_beam == target_prefix).all()

            # Copy tokens, scores and lprobs from the first beam to all beams
            tokens = self.replicate_first_beam(
                tokens, eos_mask_batch_dim, beam_size)
            scores = self.replicate_first_beam(
                scores, eos_mask_batch_dim, beam_size)
            lprobs = self.replicate_first_beam(
                lprobs, eos_mask_batch_dim, beam_size)
        return lprobs, tokens, scores

    def replicate_first_beam(self, tensor, mask, beam_size):
        """Helper function to replicate the first beam to all other beams
        according to a boolean mask."""
        tensor = tensor.view(-1, beam_size, tensor.size(-1))
        tensor[mask] = tensor[mask][:, :1, :]
        return tensor.view(-1, tensor.size(-1))

    def finalize_hypos(self, step, eos_batch_idx, eos_scores, tokens, scores,
                       finalized, finished, beam_size, attn, src_lengths,
                       max_len):
        """Finalize hypothesis, store finalized information in `finalized`, and
        change `finished` accordingly. A sentence is finalized when `beam_size`
        finished items have been collected for it.

        Returns number of sentences (not beam items) being finalized. These
        will be removed from the batch and not processed further.

        Parameters
        ----------
        step : int
            Current time step to handle.
        eos_batch_idx : torch.Tensor
            A tensor of shape (n,), where n is the number of un-finalized
            hypotheses ending in <eos>. `n` might be larger than `batch` since
            one source sentence may result in multiple hypotheses that end in
            <eos> at the same time step. The value range of this tensor is
            [0, batch * beam_size), representing the indices of the hypotheses.
        eos_scores : torch.Tensor
            A tensor of shape (n,) representing the corresponding scores of the
            hypotheses.
        tokens : torch.Tensor
            A tensor of shape (batch * beam_size, max_len + 2) storing
            generated tokens (including <sos> tokens).
        scores : torch.Tensor
            A tensor of shape (batch * beam_size, max_len + 1) storing
            accumulative beam scores.
        finalized : list
            List of finalized hypotheses's information.
        finished : list
            List of boolean values marking whether a source sentence has
            already its translated target sentence or not.
        beam_size : int
            Beam width.
        attn : torch.Tensor
            A tensor of shape (batch * beam_size, src_len, max_len + 2) (where
            `max_len + 2` can be considered tgt_len) storing attention scores
            of each token in the source sentence with each token in the
            translated sentence.
        src_lengths : torch.Tensor
            Source sentence lengths of shape (batch,).
        max_len : int
            Maximum target length.

        Returns
        -------
        newly_finished : list
            A list of newly finished sentence(s) indices (along `batch`, not
            hypotheses). The returned indices are in range [0, batch), where
            `batch` represents the number of **unfinished** sentence(s) (and it
            changes after every iteration).
        """
        assert eos_batch_idx.numel() == eos_scores.numel()

        # Clone **relevant tokens** and attention tensors.
        # tokens is (batch * beam_size, max_len). So the index_select
        # gets the newly EOS rows, then selects cols 1..{step + 2}
        # tokens_clone: (n, step + 1)
        tokens_clone = tokens.index_select(0, eos_batch_idx)[
            :, 1:step + 2
        ]  # skip the first index, which is EOS

        # The last tokens should be <eos>, which is why we are here.
        tokens_clone[:, step] = self.tgt_eos

        # Clone **relevant attention scores**
        # attn_clone: (n, src_len, step + 1)
        attn_clone = attn.index_select(0, eos_batch_idx)[
            :, :, 1:step + 2]

        # Compute scores per token position; note that `scores` already
        # started at step=1.
        # pos_scores: (n, step + 1)
        pos_scores = scores.index_select(0, eos_batch_idx)[:, :step + 1]
        pos_scores[:, step] = eos_scores

        # Convert from cumulative to per-position scores
        # pos_scores: (n, step + 1)
        pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

        # Normalize sentence-level scores
        if self.normalize_scores:
            eos_scores /= ((step + 1) ** self.len_penalty)  # (n,)

        # cum_unfin records which sentences in the batch are finished.
        # It helps match indexing between (a) the original sentences
        # in the batch and (b) the current, possibly-reduced set of
        # sentences.
        cum_unfin = []
        prev = 0
        for f in finished:
            if f:
                prev += 1
            else:
                cum_unfin.append(prev)

        # The set of tuple of (sent_idx, unfin_idx), where
        # "unfin_idx" is the index in the current (possibly reduced)
        # list of sentences, and "sent_idx" is the index in the original,
        # unreduced batch
        sents_seen = []

        # For every finished beam item
        for i in range(eos_batch_idx.size()[0]):
            idx = eos_batch_idx[i]
            score = eos_scores[i]
            # Sentence (not hypothesis) index in the current (possibly reduced)
            # batch
            unfin_idx = idx // beam_size
            # Sentence (not hypothesis) index in the original (unreduced) batch
            # Self-note: if you write down the full `finished` and the possibly
            # reduced batch (reduced = finished sentences eliminated) cum_unfin
            # you can see that this computation actually makes sense.
            sent_idx = unfin_idx + cum_unfin[unfin_idx]

            seen = (sent_idx.item(), unfin_idx.item())
            sents_seen.append(seen)

            # An input sentence (among those in a batch) is finished when
            # **enough** `beam_size` hypotheses have been collected for it
            if len(finalized[sent_idx]) < beam_size:
                hypo_attn = attn_clone[
                    i]  # (src_len, step + 1) / (src_len, tgt_len)

                finalized[sent_idx].append(
                    {
                        "tokens": tokens_clone[i],  # (step + 1,) / (tgt_len)
                        "score": score,  # scalar
                        # (src_len, step + 1) / (src_len, tgt_len)
                        "attention": hypo_attn,
                        # (step + 1,) / (tgt_len,)
                        "positional_scores": pos_scores[i],
                    }
                )

        newly_finished = []

        for sent_idx, unfin_idx in set(sents_seen):
            # Check termination conditions for this sentence
            if not finished[sent_idx]:
                num_finalized_hypos = len(finalized[sent_idx])
                assert num_finalized_hypos <= beam_size
                # If reaching terminal condition
                if num_finalized_hypos == beam_size or step == max_len:
                    finished[sent_idx] = True
                    newly_finished.append(unfin_idx)

        return newly_finished
