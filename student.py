import os
import glob
import cv2

import numpy as np
import matplotlib
from skimage.io import imread
from skimage.color import rgb2grey
from skimage.feature import hog
from skimage.transform import resize
from numpy.linalg import norm
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.spatial.distance import cdist
from scipy.stats import mode
from sklearn.svm import LinearSVC

def get_tiny_images(image_paths):
    '''
    This feature is inspired by: A. Torralba, R. Fergus, W. T. Freeman. IEEE
    Transactions on Pattern Analysis and Machine Intelligence, vol.30(11),
    pp. 1958-1970, 2008. http://groups.csail.mit.edu/vision/TinyImages/

    Inputs:
        image_paths: a 1-D Python list of strings. Each string is a complete
                     path to an image on the filesystem.
    Outputs:
        An n x d numpy array where n is the number of images and d is the
        length of the tiny image representation vector. e.g. if the images
        are resized to 16x16, then d is 16 * 16 = 256.
    '''
    images = np.zeros((len(image_paths), 256))
    for i, file in enumerate(image_paths):
        img = imread(file)
        img = rgb2grey(img)
        img = resize(img, (16, 16), anti_aliasing=True).flatten()
        images[i] = img / norm(img)
    return images

def build_vocabulary(image_paths, vocab_size):
    '''
    This function samples HOG descriptors from the training images,
    clusters them with kmeans, and then returns the cluster centers.

    Inputs:
        image_paths: a Python list of image path strings
         vocab_size: an integer indicating the number of words desired for the
                     bag of words vocab set

    Outputs:
        a vocab_size x (z*z*9) (see below) array which contains the cluster
        centers that result from the K Means clustering.
    '''


    image_list = [imread(file) for file in image_paths]

    cells_per_block = (2, 2)
    z = cells_per_block[0]
    pixels_per_cell = (4, 4)
    feature_vectors_images = []
    for image in image_list:
        feature_vectors = hog(image, feature_vector=True, pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block, visualize=False)
        feature_vectors = feature_vectors.reshape(-1, z*z*9)
        feature_vectors_images.append(feature_vectors)
    all_feature_vectors = np.vstack(feature_vectors_images)
    kmeans = MiniBatchKMeans(n_clusters=vocab_size, max_iter=500).fit(all_feature_vectors) # change max_iter for lower compute time
    vocabulary = np.vstack(kmeans.cluster_centers_)
    return vocabulary

def get_bags_of_words(image_paths):
    '''
    This function takes in a list of image paths and calculates a bag of
    words histogram for each image, then returns those histograms in an array.

    Inputs:
        image_paths: A Python list of strings, where each string is a complete
                     path to one image on the disk.

    Outputs:
        An nxd numpy matrix, where n is the number of images in image_paths and
        d is size of the histogram built for each image.
    '''

    vocab = np.load('vocab.npy')
    print('Loaded vocab from file.')
    
    vocab_length = vocab.shape[0]
    image_list = [imread(file) for file in image_paths]
    
    # Instantiate empty array
    images_histograms = np.zeros((len(image_list), vocab_length))

    cells_per_block = (2, 2) # Change for lower compute time
    z = cells_per_block[0]
    pixels_per_cell = (4, 4) # Change for lower compute time
    feature_vectors_images = []

    for i, image in enumerate(image_list):
        feature_vectors = hog(image, feature_vector=True, pixels_per_cell=pixels_per_cell,
                              cells_per_block=cells_per_block, visualize=False)
        feature_vectors = feature_vectors.reshape(-1, z*z*9)
        histogram = np.zeros(vocab_length)
        distances = cdist(feature_vectors, vocab)  
        closest_vocab = np.argsort(distances, axis=1)[:,0]
        indices, counts = np.unique(closest_vocab, return_counts=True)
        histogram[indices] += counts
        histogram = histogram / norm(histogram)
        images_histograms[i] = histogram
    return images_histograms

def svm_classify(train_image_feats, train_labels, test_image_feats):
    '''
    This function predicts a category for every test image by training
    15 many-versus-one linear SVM classifiers on the training data, then
    uses those learned classifiers on the testing data.

    Inputs:
        train_image_feats: An nxd numpy array, where n is the number of training
                           examples, and d is the image descriptor vector size.
        train_labels: An nx1 Python list containing the corresponding ground
                      truth labels for the training data.
        test_image_feats: An mxd numpy array, where m is the number of test
                          images and d is the image descriptor vector size.

    Outputs:
        An mx1 numpy array of strings, where each string is the predicted label
        for the corresponding image in test_image_feats
    '''

    clf = LinearSVC(random_state=0, tol=1e-5)
    clf.fit(train_image_feats, train_labels)
    test_predictions = clf.predict(test_image_feats)

    return test_predictions

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats):
    '''
    This function predicts the category for every test image by finding
    the training image with most similar features. For any arbitrary
    k, find the closest k neighbors and then vote among them
    to find the most common category and return that as its prediction.

    Inputs:
        train_image_feats: An nxd numpy array, where n is the number of training
                           examples, and d is the image descriptor vector size.
        train_labels: An nx1 Python list containing the corresponding ground
                      truth labels for the training data.
        test_image_feats: An mxd numpy array, where m is the number of test
                          images and d is the image descriptor vector size.

    Outputs:
        An mx1 numpy list of strings, where each string is the predicted label
        for the corresponding image in test_image_feats
    '''

    k = 5;

    # Gets the distance between each test image feature and each train image feature
    distances = cdist(test_image_feats, train_image_feats, 'euclidean')

    # Find the k closest features to each test image feature
    sorted_indices = np.argsort(distances, axis=1)
    knns = sorted_indices[:,0:k]

    # Determine the labels of those k features
    labels = np.zeros_like(knns)
    get_labels = lambda t: train_labels[t]
    vlabels = np.vectorize(get_labels)

    # Simple majority/plurality vote to choose label
    labels = vlabels(knns) # Enhance this with distances to make weighted vote
    # Pick the most common label from the k
    labels = mode(np.unique(labels, return_counts=True, axis=1)[0], axis=1)[0]

    return labels
