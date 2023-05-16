#! /usr/bin python3
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import numpy as np
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler



def run_kmeans(encoder, loader, testloader, device, n_clusters=10):
  
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    for images, labels in loader:
        train_images.append(images)
        train_labels.append(labels)
    for images, labels in testloader:
        test_images.append(images)
        test_labels.append(labels)
    train_images = torch.cat(train_images)
    train_labels = torch.cat(train_labels)
    test_images = torch.cat(test_images)
    test_labels = torch.cat(test_labels)
    with torch.no_grad():

        latent = encoder(train_images.to(device))
        latent = latent.cpu()
        # Nr. of classes/clusters
        n_digits = len(np.unique(train_labels))
        print(n_digits)

        # Initialize KMeans model
        kmeansLatent = MiniBatchKMeans(n_clusters = 10)

        # Fit the model to the training data
        kmeansLatent.fit(latent.flatten(1).cpu().numpy())

        #Assign each cluster a label by using argmax on its datapoints
        cluster_labels = infer_cluster_labels(kmeansLatent, train_labels.numpy())
        test_latent = encoder(test_images.to(device))
    

    predicted_labels = infer_data_labels(kmeansLatent.predict(test_latent.flatten(1).cpu().numpy()), cluster_labels)

    # calculate and print accuracy
    return (metrics.accuracy_score(test_labels, predicted_labels)*100)

def infer_cluster_labels(kmeans, actual_labels):
    """
    Associates most probable label with each cluster in KMeans model
    returns: dictionary of clusters assigned to each label
    """

    inferred_labels = {}

    for i in range(kmeans.n_clusters):

        # find index of points in cluster
        labels = []
        index = np.where(kmeans.labels_ == i)

        # append actual labels for each point in cluster
        labels.append(actual_labels[index])

        # determine most common label
        if len(labels[0]) == 1:
            counts = np.bincount(labels[0])
        else:
            counts = np.bincount(np.squeeze(labels))

        # assign the cluster to a value in the inferred_labels dictionary
        if np.argmax(counts) in inferred_labels:
            # append the new number to the existing array at this slot
            inferred_labels[np.argmax(counts)].append(i)
        else:
            # create a new array in this slot
            inferred_labels[np.argmax(counts)] = [i]
        
    return inferred_labels

def infer_data_labels(X_labels, cluster_labels):
    """
    Determines label for each array, depending on the cluster it has been assigned to.
    returns: predicted labels for each array
    """
    
    # empty array of len(X)
    predicted_labels = np.zeros(len(X_labels)).astype(np.uint8)
    
    for i, cluster in enumerate(X_labels):
        for key, value in cluster_labels.items():
            if cluster in value:
                predicted_labels[i] = key
                
    return predicted_labels

def KNN(encoder, loader, testloader,  device, number_of_neighbours = 5):
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    for images, labels in loader:
        train_images.append(images)
        train_labels.append(labels)
    for images, labels in testloader:
        test_images.append(images)
        test_labels.append(labels)
    train_images = torch.cat(train_images)
    train_labels = torch.cat(train_labels)
    test_images = torch.cat(test_images)
    test_labels = torch.cat(test_labels)

    clf = KNeighborsClassifier(n_neighbors=number_of_neighbours)
    train_images = encoder(train_images.to(device))
    train_images = train_images.cpu().flatten(1).detach().numpy()

    test_images = test_images.to(device)
    test_images = encoder(test_images).cpu().flatten(1).detach().numpy()
    
    scaler = MinMaxScaler()
    train_images = scaler.fit_transform(train_images)
    test_images = scaler.fit_transform(test_images)

    clf.fit(train_images,train_labels)
    
    y_pred = clf.predict(test_images)
    
    return accuracy_score(y_pred=y_pred,y_true=test_labels)*100
