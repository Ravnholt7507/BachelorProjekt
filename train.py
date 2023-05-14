import numpy as np
import torch
import torch
from tqdm import tqdm

def train(encoder, decoder, device, dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch, _ in dataloader: # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensor to the proper device
        image_batch = image_batch.to(device)
        # Encode data
        encoded_data = encoder(image_batch)
        # Decode data
        decoded_data = decoder(encoded_data)
        # Evaluate loss
        loss = loss_fn(decoded_data, image_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)

# train for one epoch to learn unique features
def train_simCLR(net, data_loader, train_optimizer, temperature, batch_size, epoch, epochs):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)

    for img_1, img_2, _ in train_bar:
        img_1, img_2 = img_1.cuda(), img_2.cuda() 
        _, out_1 = net(img_1)
        _, out_2 = net(img_2)
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0) 

        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t()) / temperature)

        # Vi masker diagoalen så vi kan sætte deres vægt til 0 og alt andet til 1 (fordi billederne er ens)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1] (Vi bruger masked_select til at sige at vi smider diagonalen væk)
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)


        # compute loss
        pair_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pair_sim = torch.cat([pair_sim, pair_sim], dim=0)

        # Final Loss
        loss = (- torch.log(pair_sim / sim_matrix.sum(dim=-1))).mean()

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num