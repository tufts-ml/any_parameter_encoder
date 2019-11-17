import torch
from torch.utils import data


def train(vae_svi, training_generator, epochs, use_cuda):
    # CUDA for PyTorch
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # cudnn.benchmark = True

    # Loop over epochs
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
        print("EPOCH {n}: {loss}".format(n=epoch, loss=epoch_loss / num_batches))

        # Validation
        # with torch.set_grad_enabled(False):
        #     for document, topics in validation_generator:
        #         # Transfer to GPU
        #         document, topics = document.to(device), topics.to(device)

        #         # Model computations
        #         [...]