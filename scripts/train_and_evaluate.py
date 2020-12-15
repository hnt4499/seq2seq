import os
import sys
import argparse
import json
import math
import datetime
from shutil import copy, SameFileError
from functools import partial
from pathlib import Path

import yaml
from loguru import logger
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from seq2seq.data import CustomDataset, collate_fn
from seq2seq.model import (Encoder, BahdanauAttention, BahdanauDecoder,
                           Seq2Seq, init_weights)
from seq2seq.training import train, evaluate
from seq2seq.utils import compare_config


DESCRIPTION = """Train and evaluate a sequence to sequence model for machine
translation."""


def main(args):
    with open(args.config_path, "r") as conf:
        config = yaml.load(conf, Loader=yaml.FullLoader)

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
    testing = training_info.get("testing", False)

    load_from = args.load_from
    resume_from = args.resume_from

    # Get save directory
    if resume_from is None:
        if work_dir is not None:
            curr_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            save_dir = os.path.join(work_dir, curr_time)
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = None
    else:
        save_dir = os.path.realpath(resume_from)
        assert os.path.exists(save_dir)

    # Get logger
    logger.remove()  # remove default handler
    logger.add(
        sys.stderr, colorize=True,
        format="<green>{time:YYYY-MM-DD at HH:mm:ss}</green> | {message}")
    if save_dir is not None:
        logger_path = os.path.join(save_dir, "training.log")
        logger.add(logger_path, mode="a",
                   format="{time:YYYY-MM-DD at HH:mm:ss} | {message}")
        logger.info(f"Working directory: {save_dir}")

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
    for dataset_name, bitext_file in bitext_files.items():
        if isinstance(bitext_file, list):
            dataloaders_ = []
            for i, bitext_f in enumerate(tqdm(bitext_file)):
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

    if load_from is not None and resume_from is not None:
        raise ValueError(
            "`load_from` and `resume_from` are mutually exclusive.")

    # Load from a pretrained model
    if load_from is not None:
        load_from = os.path.realpath(load_from)
        logger.info(f"Loading model at {load_from}")
        model.load_state_dict(torch.load(load_from, map_location=device))

    if resume_from is not None:
        # Ensure that the two configs match (with some exclusions)
        with open(os.path.join(save_dir, "config.yaml"), "r") as conf:
            resume_config = yaml.load(conf, Loader=yaml.FullLoader)
        if not compare_config(config, resume_config):
            raise RuntimeError("The two config files do not match.")

        # Load the most recent saved model
        model_list = Path(save_dir).glob("model*.pth")
        last_saved_model = max(model_list, key=os.path.getctime)
        logger.info(f"Loading most recent saved model at {last_saved_model}")
        model.load_state_dict(
            torch.load(last_saved_model, map_location=device))
        # Get some more info for resuming training
        _, last_name = os.path.split(last_saved_model)
        last_name, _ = os.path.splitext(last_name)
        _, last_epoch, last_dataloader_i = last_name.split("_")
        last_epoch, last_dataloader_i = int(last_epoch), int(last_dataloader_i)

    # Copy config
    if save_dir is not None:
        copy_from = os.path.realpath(args.config_path)
        copy_to = os.path.realpath(os.path.join(save_dir, "config.yaml"))
        try:
            copy(copy_from, copy_to)
        except SameFileError:
            pass

    # Start training and evaluating
    for epoch in range(1, num_epochs + 1):
        for dataloader_i, dataloader in enumerate(dataloaders["train"]):
            if resume_from is not None:
                if epoch < last_epoch:
                    continue
                elif epoch == last_epoch and dataloader_i <= last_dataloader_i:
                    continue
            train_loss = train(
                model, dataloader, optimizer, criterion, device, gradient_clip,
                epoch=epoch, total_epoch=num_epochs, testing=testing)
            val_loss = evaluate(
                model, dataloaders["val"], criterion, device, epoch=epoch,
                total_epoch=num_epochs, testing=testing)
            dataloader.dataset.clear()

            # Calculate perplexity
            if train_loss <= 10:
                train_ppl = f"{math.exp(train_loss):.2f}"
            else:
                train_ppl = "(...)"

            if val_loss <= 10:
                val_ppl = f"{math.exp(val_loss):.2f}"
            else:
                val_ppl = "(...)"

            # Print results
            bitext_name = os.path.split(dataloader.dataset.bitext_file)[-1]
            logger.info(
                f"Epoch: {epoch}\tFile: {bitext_name}"
                f"\tTrain loss: {train_loss:.2f}\tTrain PPL: "
                f"{train_ppl}\tVal loss: {val_loss:.2f}"
                f"\tVal PPL: {val_ppl}")

            # Save model
            if save_dir is not None:
                save_path = os.path.join(
                    save_dir, f"model_{epoch}_{dataloader_i}.pth")
                torch.save(model.state_dict(), save_path)
                logger.info(f"Model saved to {save_path}")

    # Test
    test_loss = evaluate(model, dataloaders["test"], criterion, device,
                         testing=testing)
    logger.info(
        f"Test loss: {test_loss:.4f}\tTest PPL: {math.exp(test_loss):.4f}")
    logger.info("Training finished.")


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=DESCRIPTION)
    parser.add_argument(
        '-c', '--config-path', type=str, required=True,
        help='Path to full config.')
    parser.add_argument(
        '-r', '--resume-from', type=str, required=False, default=None,
        help='Directory to resume from. Mutually exclusive with `load_from`.')
    parser.add_argument(
        '-l', '--load-from', type=str, required=False, default=None,
        help='Path to the model to load from. Mutually exclusive with '
             '`resume_from`.')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
