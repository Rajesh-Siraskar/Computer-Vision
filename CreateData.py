
# coding: utf-8

# In[12]:

# Ver 2.0: 10-Jan-2019: Add images and image file names to return statement 

# Import standard modules
import numpy as np    # N dimensional array handling
import cv2            # Open CV modules
import glob           # file handling
import random         # random shuffling
from matplotlib import pyplot as plt # for displaying images and plots

def CreateData(path_positive = 'images/training_positive_instances/', 
               path_negative = 'images/training_negative_instances/',
               file_type = '*.png', 
               limit_instances = False, display_sample_images = True):

    print('\n Create Data:')
    print('    Function will return shuffled sets of X and y data:')
    print('    X will contain image data as rows (flattened 1D numpy array)')
    print('    y will contain binary class labels: 1 for pedestrian images and 0 for backgrounds\n')

    print(' - Positive instances path: ', path_positive)
    print(' - Negative instances path: ', path_negative)

    # Import all image files and read into separate arrays
    positive_instances = []
    negative_instances = []

    # Gather all .png file names
    positive_instances = glob.glob (path_positive + file_type)
    negative_instances = glob.glob (path_negative + file_type)

    # Import all image files and read into image arrays
    n_positive = len(positive_instances)
    n_negative = len(negative_instances)

    # Initialize arrays for storing all pedestrian (positive) and background (negative) instances
    image_files = []
    labels = []
    images = []

    ## First collect all positive instances 

    # Collect all image file names 
    image_files.append([positive_instances[n] for n in range(n_positive)])

    # Next fill labels array with 'positive'
    labels.append(['positive' for n in range(n_positive)])

    ## Next collect all negative instances 

    # Collect all image file names 
    image_files.append([negative_instances[n] for n in range(n_negative)])

    # Next fill labels array with 'positive'
    labels.append(['negative' for n in range(n_negative)])

    ### Flatten lists
    labels = [p for q in labels for p in q]
    image_files = [p for q in image_files for p in q]

    # Create tuples ('class', 'image_file_name') 
    data_tuples = list(zip(labels, image_files))

    random.shuffle(data_tuples)

    # Import image files and read the files into image arrays
    n_images = len(data_tuples)
    image_files = []
    labels = []
    images = []

    # data_tuples is a list of tuples: format is ('class', 'image_file_name')

    # Separate out the class labels from data_tuples list of tuples
    labels.append([data_tuples[n][0] for n in range(n_images)])

    # Move through the data_tuples list of tuples to extract the file names
    image_files.append([data_tuples[n][1] for n in range(n_images)])

    ### Flatten list of lists

    labels = labels[0]
    image_files = image_files[0]

    y = [1 if 'positive' in label else 0 for label in labels]

    # Move through all .png files in folder and append to a local array of images
    images = np.array([np.array(cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)) 
                                for image_file in image_files])

    # Get number of instances
    n_images = len(images)
    print(' - Total instances: ', n_images)
    print(' - Positive instances: ', len(positive_instances))
    print(' - Negative instances: ', len(negative_instances))

    # If limit_instances=True, limit instances for quick experimentation
    if (limit_instances): 
        n_limited = 100

        images = images[0:n_limited]
        labels = labels[0:n_limited]
        y = y[0:n_limited]

        n_images = len(images)
        print('\n *** WARNING: Limited data-set being used ***\n')


    # Check dimensions of original images array
    print(' - NumPy Array of images created. Shape:', images.shape)

    # Flatten them to suit manipulation
    # Find number of images and it's dimensions
    n_images = images.shape[0]
    n_image_height = images.shape[1]
    n_image_width = images.shape[2]

    # Flatten 2-D images into singlular rows of data (1-D)
    X = images.flatten().reshape(n_images, n_image_height*n_image_width)

    print(' - Flattened array of images. Shape:', X.shape)

    if (display_sample_images):
        # Show first 6 images to test import  
        fig, axes = plt.subplots(1, 6, figsize=(10, 2),
                                 gridspec_kw=dict(hspace=0.1, wspace=0.1))

        for i, ax in enumerate(axes.flat):
            index = random.sample(range(n_images-1),1)[0]
            sample_image = images[index]
            class_label = labels[index]
            ax.imshow(sample_image, cmap='bone')
            ax.set(xticks=[], yticks=[], xlabel=class_label)
            if (class_label == 'positive'): ax.xaxis.label.set_color('blue')
            if (class_label == 'negative'): ax.xaxis.label.set_color('red')    

    # return the arrays
    return(images, image_files, labels, X, y)

