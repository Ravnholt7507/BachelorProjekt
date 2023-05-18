import torch
from torch import nn

class SimLoss(nn.Module):
    def __init__(self, batch_size, device):
        super(SimLoss, self).__init__()
        self.mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float().to(device)

    def forward(self, output1, output2, temp=0.7):
        batch_size = output1.shape[0]

        # out.shape = [2*B, D] where B=Batch_size and D=dimensionality
        output = torch.cat((output1, output2), dim=0)
        output_t = torch.t(output)

        # Get the similarity matrix that shows cosine similarity for each image in batch [2*B, 2*B]
        sim_matrix = torch.mm(output, output_t) / temp

        # Get all similar pairs
        similar_pairs = torch.cat([torch.diag(sim_matrix, batch_size), torch.diag(sim_matrix, -batch_size)])

        # compute loss
        pos_pairs = torch.exp(similar_pairs)
        #
        all_pairs = torch.exp(sim_matrix) * self.mask

        # Final Loss
        loss = (- torch.log(pos_pairs / all_pairs.sum(dim=-1))).mean()
        return loss
