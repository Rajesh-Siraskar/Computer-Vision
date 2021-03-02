
# coding: utf-8

# In[ ]:

# Import standard modules
import glob           # file handling
import random         # random shuffling

def SplitInstancesIntoTrainingAndTestSets(path_positive, path_negative, file_type='*.png', 
                                          training_proportion = 0.7):

    # Import all image files and read into separate arrays
    positive_instances = []
    negative_instances = []

    # Gather all .png file names
    positive_instances = glob.glob (path_positive + file_type)
    negative_instances = glob.glob (path_negative + file_type)

    # Get number of instances
    print('Positive instances: ', len(positive_instances))
    print('Negative instances: ', len(negative_instances))

    # Minimum instances for class balancing
    min_instances = min (len(positive_instances), len(negative_instances))
    print('Minimum instances for class balancing: ', min_instances)
    
    positive_instances = positive_instances[0:min_instances]
    negative_instances = negative_instances[0:min_instances]

    # Check that classes are balanced
    if (len(positive_instances) == len(negative_instances)):
        print('CHECK: Classes balanced. Number of instances in both: ', len(positive_instances))
    
    # Randomly shuffle instances
    random.shuffle(positive_instances)
    random.shuffle(negative_instances)
    
    positive_labels = min_instances*['positive']
    negative_labels = min_instances*['negative']

    class_balanced_positive_instances = list(zip(positive_labels, positive_instances))
    class_balanced_negative_instances = list(zip(negative_labels, negative_instances))
    
    # Build the complete data set by appending both lists
    data_set = class_balanced_positive_instances
    [data_set.append(ni) for ni in class_balanced_negative_instances]
    
    total_instances = len(data_set)
    print('\nSPLIT DATA SET\n- Total instances:', total_instances)
    
    # Shuffle the data
    random.shuffle(data_set)
    
    # Split the data set into training and test
    len_training = int(training_proportion * total_instances)
    training_set = data_set[0:len_training]
    test_set = data_set[len_training:total_instances]
    
    print('- Training instances:', len(training_set))
    print('- Test instances:', len(test_set))
    
    return(training_set, test_set)

