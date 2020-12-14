import os
import sys
import argparse
import json
import math
import datetime
from shutil import copy
from functools import partial

import yaml
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from rnnencdec.data import CustomDataset, collate_fn
from rnnencdec.model import (Encoder, BahdanauAttention, BahdanauDecoder,
                             Seq2Seq, init_weights)
from rnnencdec.training import train, evaluate


DESCRIPTION = """Train and evaluate a sequence to sequence model for machine
translation."""


def main(args):
    with open(args.config_path, "r") as conf:
        config = yaml.load(conf)

    # Unpack model hyperparameters
    model_info = config["model"]
    emb_dim = model_info["embedding_dim"]
    enc_hid_dim = model_info["encoder_hidden_dim"]
    dec_hid_dim = model_info["decoder_hidden_dim"]
    attn_dim = model_info["attention_dim"]
    bidirectional = model_info["bidirectional"]
    num_layers = model_info["num_layers"]
    # Unpack training hyperparameters
    training_info = config["training"]
    work_dir = training_info["work_dir"]
    dropout = training_info["dropout"]
    device = training_info["device"]
    lr = training_info["learning_rate"]
    batch_size = training_info["batch_size"]
    num_epochs = training_info["num_epochs"]
    num_workers = training_info["num_workers"]
    gradient_clip = training_info["gradient_clip"]

    # Load vocab
    src_vocab = json.load(open(config["data"]["src_vocab_path"], "r"))
    tgt_vocab = json.load(open(config["data"]["tgt_vocab_path"], "r"))
    input_dim = len(src_vocab["tok2idx"])
    output_dim = len(tgt_vocab["tok2idx"])

    # Train, val and test data
    bitext_files = config["data"]["bitext_files"]
    dataloaders = {}

    # Collate function
    collate_fn_init = partial(
        collate_fn, pad_idx=tgt_vocab["tok2idx"]["<pad>"])

    num_samples = config["data"].get(
        "num_samples", [None] * len(bitext_files["train"]))
    if isinstance(batch_size, int):
        batch_size = [batch_size] * len(bitext_files["train"])
    assert len(num_samples) == len(batch_size) == len(bitext_files["train"])

    # Initialize dataloaders
    for dataset_name, bitext_file in tqdm(bitext_files.items()):
        if isinstance(bitext_file, list):
            dataloaders_ = []
            for i, bitext_f in enumerate(bitext_file):
                dataset = CustomDataset(
                    bitext_f, src_vocab, tgt_vocab,
                    num_samples=num_samples[i], shuffle=True)
                dataloader = DataLoader(
                    dataset, batch_size=batch_size[i], shuffle=False,
                    collate_fn=collate_fn_init, num_workers=num_workers)
                dataloaders_.append(dataloader)
            dataloaders[dataset_name] = dataloaders_
        else:
            dataset = CustomDataset(bitext_file, src_vocab, tgt_vocab,
                                    num_samples=None, shuffle=True)
            dataloader = DataLoader(
                dataset, batch_size=batch_size[-1], shuffle=False,
                collate_fn=collate_fn_init, num_workers=num_workers)
            dataloaders[dataset_name] = dataloader

    # Initialize model
    device = torch.device(device)
    encoder = Encoder(input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout,
                      num_layers, bidirectional)
    attention = BahdanauAttention(enc_hid_dim, dec_hid_dim, attn_dim,
                                  bidirectional)
    decoder = BahdanauDecoder(output_dim, emb_dim, enc_hid_dim, dec_hid_dim,
                              dropout, attention, bidirectional)
    model = Seq2Seq(encoder, decoder, device).to(device)

    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab["tok2idx"]["<pad>"])

    # Save to a directory
    curr_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(work_dir, curr_time)
    os.makedirs(save_dir, exist_ok=True)
    # Copy config
    copy_from = os.path.realpath(args.config_path)
    copy_to = os.path.realpath(os.path.join(save_dir, "config.yaml"))
    copy(copy_from, copy_to)

    # Start training and evaluating
    for epoch in range(num_epochs):
        for dataloader in dataloaders["train"]:
            train_loss = train(model, dataloader, optimizer, criterion, device,
                               gradient_clip)
            val_loss = evaluate(model, dataloaders["val"], criterion, device)
            dataloader.dataset.clear()

        print(f"Epoch: {epoch + 1}\tTrain loss: {train_loss:.2f}\tTrain PPL: "
              f"{math.exp(train_loss):.2f}\tVal loss: {val_loss:.2f}"
              f"\tVal PPL: {math.exp(val_loss):.2f}")

        # Save model
        save_path = os.path.join(save_dir, f"model_{epoch}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    # Test
    test_loss = evaluate(model, dataloaders["test"], criterion, device)
    print(f"Test loss: {test_loss:.4f}\tTest PPL: {math.exp(test_loss):.4f}")


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=DESCRIPTION)
    parser.add_argument(
        '-c', '--config-path', type=str, required=True,
        help='Path to full config.')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
