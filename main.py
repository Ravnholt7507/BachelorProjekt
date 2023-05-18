from dataloaders import simclr_dataloader, normal_loader, get_cifar
from models.SimCLR import Model
from models.CAE import Encoder, Decoder
from train import train_simCLR, train_autoencoder
from classifier import KNN, run_kmeans
import torch 
from torch import nn
from t_SNE import tSNE
from tqdm import tqdm

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data, _, _ = simclr_dataloader()
    fit_loader, test_loader = normal_loader(batch_size=128)
  #  model = Model().to(device)
    encoder = Encoder()
    decoder = Decoder()

    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(params_to_optimize, lr=0.001)

    losses = []
    epochs = 2
    for epoch in range(epochs):
       # losses.append(train_simCLR(model, device, train_data, optimizer, 0.7, 128, epoch, epochs))
         losses.append(train_autoencoder(encoder, decoder, device, fit_loader, loss_fn, optimizer, epoch, epochs))

    tSNE(encoder, get_cifar(), device=device)

    print(f"score: {KNN(encoder, fit_loader, test_loader, device, number_of_neighbours=5)}%")

