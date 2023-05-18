import torch
from sklearn.manifold import TSNE
import plotly.express as px
import numpy as np
import pandas as pd


def tSNE(encoder, test_dataset, device):
    encoded_samples = []

    for sample in test_dataset:
        img = sample[0].unsqueeze(0).to(device)
        label = sample[1]
        # Encode image
        encoder.eval()
        with torch.no_grad():
            encoded_img = encoder(img)
        # Append to list
        encoded_img = encoded_img.flatten().cpu().numpy()
        encoded_sample = {f"Enc. Variable {i}": enc for i, enc in enumerate(encoded_img)}
        encoded_sample['label'] = label
        encoded_samples.append(encoded_sample)
    encoded_samples = pd.DataFrame(encoded_samples)

    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(encoded_samples.drop(['label'], axis=1))
    fig = px.scatter(tsne_results, x=0, y=1,
                     color=encoded_samples.label.astype(str),
                     labels={'0': 'tsne-2d-one', '1': 'tsne-2d-two'})
    fig.show()
