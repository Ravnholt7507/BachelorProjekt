# BachelorProjekt:
When the repository is pulled, an example version of a main.py is already included. This example is using the SimCLR model, on the Cifar10 dataset, with KNN as a classifier.

## The dataloaders:
there are two different dataloaders implented.
first off we have the "normal_dataloader", which returns two lists of images and labels, the first list is for training and the second list is for testing. Both lists has the shape [images, labels]. This dataloader is intended for use with the autoencoder and the convolutional autoencoder, but can also be used like we have in the example in the main.py file.

The second dataloader is called "simCLR_dataloader", this returns 3 lists, one for training, one fore testing, and one for memory when the weighted KNN is used. all lists has the form [transformed_image1, transformed_image2, target_image]

Both dataloaders have a standard batchsize of 128.

## The models:
We currently have three different models, an autoencoder, a convolutional autoencoder and SimCLR. The autoencoder and the convolutional autoencoder both have an Encoder class and a Decoder class. The models are quite small at the moment, as there is no need for it to be bigger when it is used on the Cifar10 dataset. The model can however easily be expanded in the code if needed.

The SimCLR model only contain one class called Model. This model uses a ResNet18 model, that we have modified a bit in order to make it fit on the Cifar10 dataset. The model has a standard feature dimensionality of 128.

## The training:
In the train.py file, we have two different functions for training the models.

If the model used is an autoencoder or a convolutional autoencoder the train function should be used, which takes the inputs: {encoder model}, {decoder model}, {CUDA or CPU?}, {dataset}, {loss function}, {optimizer parameters}, {current epoch} and {all epochs}. The function outputs the avereage train loss from the entire epoch.

If the model used is SimCLR, the train_simclr function should be used. This function takes the inputs: {model}, {dataset}, {optimizer parameters}, {temperature},  {batch_size}, {current epoch} and {all epochs}. The function outputs the average loss for a batch in that epoch.

## The classifier:
There are three built in classifiers at the moment. The k-kmeans and KNN classifier can be used with all models.
Both classifiers takes the inputs: {Model}, {training dataset}, {test dataset}, {CUDA or CPU} and {number of clusters/neighbours}. The standard number of clusters for k-means is 10, and the standard number of neighbours for KNN is 5.

For SimCLR we also have a weighted KNN classifier, which achieves a bit better results than the ordinary KNN with the SimCLR model. This function take the inputs: {model}, {memory dataset} and {test dataset}

## Explanation of example:
In the example we use two different dataloader, one called "simclr_dataloader" and another called "normal_loader". We do this because the KNN classifier requires dataloader input of the form [image, label], but the SimCLR_dataloader has the form [image1, image2, target], and is used for training the simCLR model. 
