import numpy as np
from losses import SimLoss
import torch
from tqdm import tqdm


def train_autoencoder(encoder, decoder, device, dataloader, loss_fn, optimizer, epoch, epochs, ae):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    trainbar = tqdm(dataloader)
    if ae == True:
        for imagebatch, _ in trainbar:
            for image in imagebatch:
                image = image.to(device)
                image = image.reshape(-1, 3*32*32)
                # Encode data
                encoded_data = encoder(image)
                # Decode data
                decoded_data = decoder(encoded_data)
                # Evaluate loss
                loss = loss_fn(decoded_data, image)
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Print batch loss
                train_loss.append(loss.detach().cpu().numpy())
                trainbar.set_description(f"Train Epoch: [{epoch}/{epochs}] Loss: {np.mean(train_loss):.4f}")
    else:
        # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
        for image_batch, _ in trainbar:
            # with "_" we just ignore the labels (the second element of the dataloader tuple)
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
            trainbar.set_description(f"Train Epoch: [{epoch}/{epochs}] Loss: {np.mean(train_loss):.4f}")

    return np.mean(train_loss)


# train for one epoch to learn unique features
def train_simCLR(net, device, data_loader, train_optimizer, temp, batch_size, epoch, epochs):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    criterion = SimLoss(batch_size, device)

    for img_1, img_2, _ in train_bar:
        img_1, img_2 = img_1.to(device), img_2.to(device)
        _, out_1 = net(img_1)
        _, out_2 = net(img_2)

        loss = criterion(out_1, out_2, temp)

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch + 1, epochs, total_loss / total_num))

    return total_loss / total_num
