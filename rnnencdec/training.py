"""
References
1. PyTorch tutorial: Language translation with torchtext.
   https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html
"""


import math

from tqdm import tqdm
import torch


def train(model, dataloader, optimizer, criterion, device, gradient_clip=None,
          epoch=None, total_epoch=None):
    """Train the model.

    Parameters
    ----------
    model : nn.Module
        Initialized model.
    dataloader : torch.utils.data.DataLoader
        Training data loader.
    optimizer : torch.optim
        Initialized optimizer for the model.
    criterion : nn.Module
        Loss (objective) function.
    device : torch.device
        Device in which computation is performed.
    gradient_clip : float
        Value to clip the gradient. If None, gradient clipping will not be
        performed.
    epoch : int
        Current epoch index. Used to print meaningful progress bar with tqdm if
        specified.
    total_epoch : int
        Total number of epochs. Used to print meaningful progress bar with tqdm
        if specified.
    """
    model.train()
    epoch_loss = 0

    with tqdm(dataloader) as t:
        if epoch is not None and total_epoch is not None:
            t.set_description(f"Training ({epoch}/{total_epoch})")
        else:
            t.set_description("Training")
        for i, (src, tgt) in enumerate(t):
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()

            # Forward
            output = model(src, tgt)

            # Compute loss
            output = output[1:].view(-1, output.shape[-1])
            tgt = tgt[1:].view(-1)
            loss = criterion(output, tgt)

            # Backward
            loss.backward()
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip)
            optimizer.step()

            epoch_loss += loss.item()
            t.set_postfix(loss=f"{loss.item():.4f}",
                          acc_loss=f"{(epoch_loss / (i + 1)):.4f}")

    return epoch_loss / (len(dataloader) + 1)


def evaluate(model, dataloader, criterion, device, epoch=None,
             total_epoch=None):
    """Evaluate the trained model.

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
    epoch : int
        Current epoch index. Used to print meaningful progress bar with tqdm if
        specified.
    total_epoch : int
        Total number of epochs. Used to print meaningful progress bar with tqdm
        if specified.
    """
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        with tqdm(dataloader) as t:
            if epoch is not None and total_epoch is not None:
                t.set_description(f"Evaluating ({epoch}/{total_epoch})")
            else:
                t.set_description("Evaluating")
            for i, (src, tgt) in enumerate(t):
                src, tgt = src.to(device), tgt.to(device)

                # Forward
                output = model(src, tgt, 0)  # turn off teacher forcing

                # Compute loss
                output = output[1:].view(-1, output.shape[-1])
                tgt = tgt[1:].view(-1)
                loss = criterion(output, tgt)

                # Update
                epoch_loss += loss.item()
                t.set_postfix(loss=f"{loss.item():.4f}",
                              acc_loss=f"{(epoch_loss / (i + 1)):.4f}")

    return epoch_loss / (len(dataloader) + 1)
