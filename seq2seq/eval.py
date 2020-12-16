"""
References
1. PyTorch tutorial: Language translation with torchtext.
   https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html
"""


from tqdm import tqdm
import torch


def evaluate(model, dataloader, criterion, device, epoch=None,
             total_epoch=None, testing=False):
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
    testing : bool
        If True, only run for 10 iterations. Useful for debugging and finding
        batch sizes, etc. (default: False)
    """
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        total = 10 if testing else len(dataloader)
        with tqdm(dataloader, total=total, leave=False) as t:
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
                t.set_postfix(loss=f"{loss.item():.2f}",
                              acc_loss=f"{(epoch_loss / (i + 1)):.2f}")

                # Break when reaching 10 iterations when testing
                if testing and i == 9:
                    break

    return epoch_loss / (i + 1)
