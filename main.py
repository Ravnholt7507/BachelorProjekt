from dataloaders import simclr_dataloader, normal_loader, get_cifar
from models.SimCLR import Model
from models.autoencoder import Encoder, Decoder
from train import train_simCLR, train_autoencoder
from classifier import KNN, run_kmeans
import torch 
from torch import nn
from t_SNE import tSNE
from tqdm import tqdm

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fit_loader, test_loader = normal_loader(batch_size=128)
  #  model = Model().to(device)
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)

    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(params_to_optimize, lr=0.001)

    losses = []
    epochs = 10
    for epoch in range(epochs):
         losses.append(train_autoencoder(encoder, decoder, device, fit_loader, loss_fn, optimizer, epoch, epochs, ae=True))

    print(f"score: {KNN(encoder, fit_loader, test_loader, device, number_of_neighbours=5)}%")

