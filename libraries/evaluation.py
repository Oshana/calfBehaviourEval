from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import cohen_kappa_score


def get_class_wise_accuracies(matrix): 
    """
    Calculate class-wise accuracies from the confusion matrix.

    Args:
    confusion_matrix: The confusion matrix where
        confusion_matrix[i][j] is the number of instances of class i predicted as class j.

    Returns:
    dict: A dictionary where keys are class indices and values are the accuracy
        of each class.
    """
    diag = matrix.diagonal()
    sums = matrix.sum(axis=1)

    class_wise_accuracies = {}
    for i in range(len(sums)):
        if sums[i] == 0:
            class_wise_accuracies[i] = 0
        else:
            class_wise_accuracies[i] = diag[i]/sums[i]
            
    return class_wise_accuracies


def evaluate_model(y_test, y_pred):
    
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    macro_precision = precision_score(y_test, y_pred, average='macro', zero_division = 0)
    macro_recall = recall_score(y_test, y_pred, average='macro', zero_division = 0)
    macro_f1_score = f1_score(y_test, y_pred, average='macro', zero_division = 0)
    class_precisions = precision_score(y_test, y_pred, average=None, zero_division = 0)
    class_recalls = recall_score(y_test, y_pred, average=None, zero_division = 0)
    class_f1_scores = f1_score(y_test, y_pred, average=None, zero_division = 0)
    classification_report_ = classification_report(y_test, y_pred)
    confusion_matrix_ = confusion_matrix(y_test, y_pred)
    class_wise_accuracies = get_class_wise_accuracies(confusion_matrix_)
    cohen_kappa = cohen_kappa_score(y_test, y_pred)
    
    return {
        'balanced_accuracy': balanced_accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1_score': macro_f1_score,
        'class_precisions': class_precisions,
        'class_recalls': class_recalls,
        'class_f1_scores': class_f1_scores,
        'class_wise_accuracies': class_wise_accuracies,
        'classification_report': classification_report_,
        'confusion_matrix': confusion_matrix_,
        'cohen_kappa': cohen_kappa
    }