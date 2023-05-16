from dataloaders import simclr_dataloader, normal_loader
from models.SimCLR import Model
from train import train_simCLR
from classifier import KNN, run_kmeans
import torch 
from torch import nn
from tqdm import tqdm

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data, _, _= simclr_dataloader()
    fit_loader, test_loader = normal_loader(batch_size=4)
    model = Model().to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.001)

    losses = []
    epochs = 2
    for epoch in range(epochs):
        losses.append(train_simCLR(model, train_data, optimizer, 0.7, 128, epoch, epochs))

    print(f"score: {KNN(model, fit_loader, test_loader, 5, device)}%")