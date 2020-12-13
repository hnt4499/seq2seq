"""
References:
1. PyTorch tutorial: Language translation with torchtext.
   https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html
"""


import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """General-purpose encoder.

    Parameters
    ----------
    input_dim : int
        Input vocab size.
    emb_dim : int
        Dimension of the word embeddings.
    enc_hid_dim : int
        Dimension of the encoder hidden units.
    dec_hid_dim : int
        Dimension of the decoder hidden units. Used to generate representation
        vector that is compatible with the decoder.
    dropout : float
        Dropout for word embeddings.
    num_layers : int
        Number of layers of the RNN model. (default: 1)
    bidirectional : bool
        Whether to use bidirectional RNN model. (default: True)
    """
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout,
                 num_layers=1, bidirectional=True, emb_mat=None):
        super(Encoder, self).__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        # Embedding layer
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(input_dim, emb_dim)

        # RNN layer (GRU)
        self.rnn = nn.GRU(input_size=emb_dim, hidden_size=enc_hid_dim,
                          num_layers=num_layers, bidirectional=bidirectional)
        # FC layer
        num_directions = 2 if bidirectional else 1
        self.fc = nn.Linear(num_directions * num_layers * enc_hid_dim,
                            dec_hid_dim)
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        """Forward step.

        Parameters
        ----------
        src : torch.Tensor
            Input sentence of shape (seq_len, batch)

        Returns
        -------
        outputs : torch.Tensor
            The last RNN layer's hidden states at all time steps. Shaped
            (seq_len, batch, num_directions * enc_hid_dim).
        hidden : torch.Tensor
            The hidden states at the last time steps of all RNN layers. Shaped
            (batch, dec_hid_dim).
        """
        # (seq_len, batch, emb_dim)
        embedded = self.dropout(self.embedding(src))

        # (seq_len, batch, num_directions * enc_hid_dim) and
        # (num_layers * num_directions, batch, enc_hid_dim)
        outputs, hidden = self.rnn(embedded)

        # Turn encoder's last hidden state to a vector to feed to the decoder
        hidden = [hidden[i, :, :] for i in range(hidden.shape[0])]  # unpack

        # (batch, enc_hid_dim * num_layers * num_directions)
        hidden = torch.cat(hidden, dim=1)
        hidden = torch.tanh(self.fc(hidden))  # (batch, dec_hid_dim)

        return outputs, hidden


class BahdanauAttention(nn.Module):
    """Bahdanau attention as described in the paper
        Bahdanau, D., Cho, K., & Bengio, Y. (2016). Neural Machine Translation
        by Jointly Learning to Align and Translate. ArXiv:1409.0473 [Cs, Stat].


    Parameters
    ----------
    enc_hid_dim : int
        Dimension of the encoder hidden units.
    dec_hid_dim : int
        Dimension of the decoder hidden units.
    attn_dim : int
        Dimension of the attention vectors.
    bidirectional : bool
        Whether bidirectional RNN was used in the encoder. (default: True)
    """
    def __init__(self, enc_hid_dim, dec_hid_dim, attn_dim, bidirectional=True):
        super(BahdanauAttention, self).__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        num_directions = 2 if bidirectional else 1
        self.attn_in = num_directions * enc_hid_dim + dec_hid_dim

        self.attn = nn.Linear(self.attn_in, attn_dim)

    def forward(self, decoder_hidden, encoder_outputs):
        """Forward step.

        Parameters
        ----------
        decoder_hidden : torch.Tensor
            Previous decoder hidden state of shape (batch, dec_hid_dim).
        encoder_outputs : torch.Tensor
            Encoder outputs (first element of the returned tuple) containing
            all last layer's hidden states of shape
            (seq_len, batch, num_directions * enc_hid_dim).

        Returns
        -------
        attention : torch.Tensor
            A vector which represents the probabilities of attending to a word
            over all word in the input sentence.
        """
        src_len = encoder_outputs.shape[0]

        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(
            1, src_len, 1)  # (batch, seq_len, dec_hid_dim)

        encoder_outputs = encoder_outputs.permute(
            1, 0, 2)  # (batch, seq_len, num_directions * enc_hid_dim)

        # Compute "energy"
        # (batch, seq_len, num_directions * enc_hid_dim + dec_hid_dim)
        energy = torch.cat((repeated_decoder_hidden, encoder_outputs), dim=2)
        energy = torch.tanh(self.attn(energy))  # (batch, seq_len, attn_dim)

        # Turn into probabilities of shape # (batch, seq_len)
        attention = F.softmax(torch.sum(energy, dim=2), dim=1)

        return attention


class BahdanauDecoder(nn.Module):
    """Decoder for the Bahdanau attention model as described in
        Bahdanau, D., Cho, K., & Bengio, Y. (2016). Neural Machine Translation
        by Jointly Learning to Align and Translate. ArXiv:1409.0473 [Cs, Stat].
    Note that this implementation is slightly different from the original
    architecture described in the paper, where the predictions are computed by
    a softmax layer rather than a maxout layer.

    Parameters
    ----------
    output_dim : int
        Output vocab size.
    emb_dim : int
        Dimension of the word embeddings.
    enc_hid_dim : int
        Dimension of the encoder hidden units.
    dec_hid_dim : int
        Dimension of the decoder hidden units.
    dropout : float
        Dropout for word embeddings.
    attention : nn.Module
        Initialized attention module.
    bidirectional : bool
        Whether bidirectional RNN was used in the encoder. (default: True)
    """
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout,
                 attention, bidirectional=True):
        super(BahdanauDecoder, self).__init__()

        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        num_directions = 2 if bidirectional else 1
        self.rnn = nn.GRU(emb_dim + num_directions * enc_hid_dim, dec_hid_dim)

        self.out = nn.Linear(self.attention.attn_in + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def _weighted_encoder_rep(self, decoder_hidden, encoder_outputs):
        """Compute weighted encoder representation.

        Parameters
        ----------
        decoder_hidden : torch.Tensor
            Current decoder hidden state of shape (batch, dec_hid_dim).
        encoder_outputs : torch.Tensor
            Encoder outputs (first element of the returned tuple) containing
            all last layer's hidden states of shape
            (seq_len, batch, num_directions * enc_hid_dim).

        Returns
        -------
        weighted_encoder_rep : torch.Tensor
            Weighted encoder representation of shape
            (1, batch, num_direction * enc_hid_dim) where `num_directions` is
            the number of directions of the RNN layer in the encoder.
        """

        a = self.attention(decoder_hidden, encoder_outputs)  # (batch, seq_len)

        a = a.unsqueeze(1)  # (batch, 1, seq_len)

        encoder_outputs = encoder_outputs.permute(
            1, 0, 2)  # (batch, seq_len, num_directions * enc_hid_dim)

        weighted_encoder_rep = torch.bmm(
            a, encoder_outputs)  # (batch, 1, num_directions * enc_hid_dim)

        weighted_encoder_rep = weighted_encoder_rep.permute(
            1, 0, 2)  # (1, batch, num_directions * enc_hid_dim)

        return weighted_encoder_rep  # (1, batch, num_directions * enc_hid_dim)

    def forward(self, input, decoder_hidden, encoder_outputs):
        """Forward step.

        Parameters
        ----------
        input : torch.Tensor
            Input of the decoder at a time step of shape (batch,).
        decoder_hidden : torch.Tensor
            Current decoder hidden state of shape (batch, dec_hid_dim).
        encoder_outputs : torch.Tensor
            Encoder outputs (first element of the returned tuple) containing
            all last layer's hidden states of shape
            (seq_len, batch, num_directions * enc_hid_dim).

        Returns
        -------
        output : torch.Tensor
            Prediction of shape (batch, output_dim).
        decoder_hidden : torch.Tensor
            Current decoder hidden state of shape (batch, dec_hid_dim).
        """

        input = input.unsqueeze(0)  # (1, batch)

        embedded = self.dropout(self.embedding(input))  # (1, batch, emb_dim)

        # (1, batch, num_directions * enc_hid_dim)
        weighted_encoder_rep = self._weighted_encoder_rep(
            decoder_hidden, encoder_outputs)

        rnn_input = torch.cat(
            (embedded, weighted_encoder_rep),
            dim=2)  # (1, batch, emb_dim + num_directions * enc_hid_dim)

        # both are (1, batch, dec_hid_dim)
        output, decoder_hidden = self.rnn(
            rnn_input, decoder_hidden.unsqueeze(0))

        embedded = embedded.squeeze(0)  # (batch, emb_dim)
        output = output.squeeze(0)  # (batch, dec_hid_dim)
        weighted_encoder_rep = weighted_encoder_rep.squeeze(
            0)  # (batch, num_directions * enc_hid_dim)

        # (batch, num_directions * enc_hid_dim + dec_hid_dim + emb_dim)
        output = torch.cat((output, weighted_encoder_rep, embedded), dim=1)
        output = self.out(output)  # (batch, output_dim)

        return output, decoder_hidden.squeeze(0)


class Seq2Seq(nn.Module):
    """Decoder for the Bahdanau attention model as described in
        Bahdanau, D., Cho, K., & Bengio, Y. (2016). Neural Machine Translation
        by Jointly Learning to Align and Translate. ArXiv:1409.0473 [Cs, Stat].

    Parameters
    ----------

    """
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        """Forward step.

        Parameters
        ----------
        src : torch.Tensor
            Input sentence of shape (seq_len, batch).
        tgt : torch.Tensor
            Expected output sentence of shape (seq_len, batch).
        teacher_forcing_ratio : torch.Tensor
            Probability of forcing true input to the encoder. (default: 0.5)

        Returns
        -------
        outputs : torch.Tensor
            Output predictions at every timestep of shape
            (seq_len, batch_size, output_dim).
        """

        batch_size = src.shape[1]
        max_len = tgt.shape[0]
        tgt_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, tgt_vocab_size).to(
            self.device)  # (seq_len, batch_size, output_dim)

        # (seq_len, batch, num_directions * enc_hid_dim) and
        # (batch, dec_hid_dim)
        encoder_outputs, hidden = self.encoder(src)

        # First input to the decoder is the <sos> token
        output = tgt[0, :]  # (batch,)

        for t in range(1, max_len):
            # (batch, output_dim) and (batch, dec_hid_dim)
            output, hidden = self.decoder(
                output, hidden, encoder_outputs)

            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (tgt[t] if teacher_force else top1)

        return outputs


def init_weights(m):
    """Weight initialization"""
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
