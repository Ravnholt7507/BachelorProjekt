# BachelorProjekt:
When the repository is pulled, an example version of a main.py is already included. This example is using the SimCLR model, on the Cifar10 dataset, with KNN as a classifier.

## Explanation of example:
In the example we use two different dataloader, one called "simclr_dataloader" and another called normal_loader. We do this because the KNN classifier requires dataloader input of the form [image, label], but the SimCLR_dataloader has the form [image1, image2, target], it is done like this because we want to be able to use the wieghted KNN as a classifier aswell. 

## The dataloaders:
we currently have two different dataloaders implented.
first off we have the "normal_dataloader", which returns two lists of images and labels, the first list is for training and the second list is for testing. Both lists has the shape [images, labels]. This dataloader is intended for use with the autoencoder and the conveluted autoencoder, but can also be used like we have in the example in the main.py file.

The second dataloader is called "simCLR_dataloader", this returns 3 lists, one for training, one fore testing, and one for memory when the weighted KNN is used. all lists has the form [transformed_image1, transformed_image2, target_image]

## The models:
placeholder

## The training:
placeholder

## The classifier:
placeholder
