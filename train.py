import torch
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter


def train(vae_svi, training_generator, validation_generator, epochs=1, use_cuda=True, results_dir=None, name=''):
    """ Specifically to train APE using topics/document combinations """
    writer = SummaryWriter(results_dir)
    # CUDA for PyTorch
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # cudnn.benchmark = True

    # Loop over epochs
    step = 0
    for epoch in range(epochs):
        epoch_loss = 0
        # Training
        num_batches = 0
        for document, topics in training_generator:
            # Transfer to GPU
            if use_cuda:
                document, topics = document.to(device), topics.to(device)
            # Model computations
            epoch_loss += vae_svi.step(document, topics)
            num_batches += 1
            step += 1
            writer.add_scalar(f'{name} training loss', epoch_loss / num_batches, step)
            if step % 30 == 0:
                # Validation
                with torch.set_grad_enabled(False):
                    val_loss = 0
                    num_val_batches = 0
                    for document, topics in validation_generator:
                        # Transfer to GPU
                        if use_cuda:
                            document, topics = document.to(device), topics.to(device)
                        val_loss += vae_svi.evaluate_loss(document, topics)
                        num_val_batches += 1
                writer.add_scalar(f'{name} validation loss', val_loss / num_val_batches, step)
    return vae_svi


def train_from_scratch(vae_svi, training_generator, validation_generator, epochs=1, use_cuda=True, results_dir=None, name=''):
    writer = SummaryWriter(results_dir)
    # CUDA for PyTorch
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # cudnn.benchmark = True

    # Loop over epochs
    step = 0
    for epoch in range(epochs):
        epoch_loss = 0
        # Training
        num_batches = 0
        for document in training_generator:
            # Transfer to GPU
            if use_cuda:
                document = document.to(device)
            # Model computations
            epoch_loss += vae_svi.step(document)
            num_batches += 1
            step += 1
            writer.add_scalar(f'{name} training loss', epoch_loss / num_batches, step)
            if step % 30 == 0:
                # Validation
                with torch.set_grad_enabled(False):
                    val_loss = 0
                    num_val_batches = 0
                    for document in validation_generator:
                        # Transfer to GPU
                        if use_cuda:
                            document = document.to(device)
                        val_loss += vae_svi.evaluate_loss(document)
                        num_val_batches += 1
                writer.add_scalar(f'{name} validation loss', val_loss / num_val_batches, step)
    return vae_svi