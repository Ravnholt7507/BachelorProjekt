from dataloaders import normal_loader
from models.CAE import Encoder, Decoder
from train import train
from classifier import KNN, run_kmeans
import torch 
from torch import nn
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loader, testloader = normal_loader()
encoder = Encoder().to(device)
decoder = Decoder().to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

losses = []
epochs = 2
trainbar = tqdm(range(epochs))
for epoch in trainbar:
    losses.append(train(encoder, decoder, device, loader, loss_fn, optimizer))

print(f"score: {run_kmeans(encoder=encoder, loader=loader, testloader=loader, device=device)}%")