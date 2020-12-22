"""
References
1. PyTorch tutorial: Language translation with torchtext.
   https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html
"""


from tqdm import tqdm
import torch


def train(model, dataloader, optimizer, criterion, device, gradient_clip=None,
          epoch=None, total_epoch=None, testing=False, evaluate_every=None,
          evaluator=None):
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
    testing : bool
        If True, only run for 10 iterations. Useful for debugging and finding
        batch sizes, etc. (default: False)
    evaluate_every : int
        If not None, perform evaluate every `evaluate_every` iterations during
        training.
    evaluator : callable
        A (possibly partially initialized) evaluator that performs evaluation
        when called with `model` (i.e., `evaluator(model)`).
    """
    model.train()
    epoch_loss = 0

    total = 10 if testing else len(dataloader)
    with tqdm(dataloader, total=total, leave=False) as t:
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

            if evaluate_every is not None and (i + 1) % evaluate_every == 0:
                if evaluator is not None:
                    evaluator(
                        model=model,
                        prefix=f"[Validation (batch: {i + 1}/{total})] ")

            # Break when reaching 10 iterations when testing
            if testing and i == 9:
                break

    return epoch_loss / (i + 1)
