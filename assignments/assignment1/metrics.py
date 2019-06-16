import numpy as np
def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    conf_matrix = np.zeros((2, 2))
    for i in range(len(prediction)):
        if prediction[i] and ground_truth[i]:
            conf_matrix[0][0] += 1
        elif not prediction[i] and not ground_truth[i]:
            conf_matrix[1][1] += 1
        elif not prediction[i] and ground_truth[i]:
            conf_matrix[1][0] += 1
        else:
            conf_matrix[0][1] += 1

    print('Confusion matrix = ', conf_matrix)

    precision = conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[0][1])
    recall = conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[1][0])
    accuracy = (conf_matrix[0][0] + conf_matrix[1][1])/conf_matrix.sum()
    f1 = 2 * precision * recall/ (precision + recall)


    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    return 0
