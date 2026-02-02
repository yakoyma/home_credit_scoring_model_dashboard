"""
===============================================================================
This file contains all the functions for the project
===============================================================================
"""
# Libraries
import matplotlib.pyplot as plt
import pandas as pd


from csv import Sniffer
from sklearn.metrics import (fbeta_score,
                             accuracy_score,
                             balanced_accuracy_score,
                             f1_score,
                             confusion_matrix,
                             classification_report,
                             ConfusionMatrixDisplay,
                             roc_auc_score,
                             PrecisionRecallDisplay,
                             roc_curve,
                             auc)
from imblearn.metrics import classification_report_imbalanced



def load_dataset(file_path, encoding):
    """This function loads a csv file and finds the type of separators by
     sniffing the file.

    Args:
        file_path (str): the csv file path
        encoding (str): encoding to use for reading and writing

    Returns:
        dataset (pd.DataFrame): the loaded Pandas dataset
    """

    with open(file_path, 'r') as csvfile:
        separator = Sniffer().sniff(csvfile.readline()).delimiter
    dataset = pd.read_csv(
        filepath_or_buffer=file_path,
        sep=separator,
        encoding=encoding,
        encoding_errors='ignore',
        on_bad_lines='skip'
    )
    return dataset


def dataset_info_description(dataset, max_rows):
    """This function displays the information and description of a Pandas
    DataFrame.

    Args:
        dataset (pd.DataFrame): the Pandas DataFrame
        max_rows (int): the maximum number of rows in the dataset to be
                        displayed
    """

    # Display dimensions of the dataset
    print('\nDimensions of the dataset: {}'.format(dataset.shape))

    # Display information about the dataset
    print('\nInformation about the dataset:')
    print(dataset.info())

    # Display the description of the dataset
    print('\nDescription of the dataset:')
    print(dataset.describe(include='all'))

    # Display the head and the tail of the dataset
    print('\nDisplay the head and the tail of the dataset: ')
    print(pd.concat([dataset.head(max_rows), dataset.tail(max_rows)]))


def evaluate_binary_classification(y_test, y_pred, y_proba, beta, labels):
    """This function evaluates the result of a Binary Classification.

    Args:
        y_test (array-like): the test labels
        y_pred (array-like): the predicted labels
        y_proba (array-like): the predicted probabilities
        beta (int): the ratio of recall importance to precision importance
                    of F-beta score
        labels (array-like): list of unique labels for Confusion Matrix Plot
    """

    print('\n\nF-beta: {:.3f}'.format(fbeta_score(
        y_true=y_test, y_pred=y_pred, beta=beta, labels=labels)))
    print('Accurcay: {:.3f}'.format(accuracy_score(
        y_true=y_test, y_pred=y_pred)))
    print('Balanced Accurcay: {:.3f}'.format(balanced_accuracy_score(
        y_true=y_test, y_pred=y_pred)))
    print('F1 score: {:.3f}'.format(f1_score(
        y_true=y_test, y_pred=y_pred)))
    print('Confusion Matrix:\n{}'.format(confusion_matrix(
        y_true=y_test, y_pred=y_pred)))
    print('Classification Report:\n{}'.format(classification_report(
        y_true=y_test, y_pred=y_pred)))
    print('Imblearn Classification Report:\n{}'.format(
        classification_report_imbalanced(y_true=y_test, y_pred=y_pred)))
    display = ConfusionMatrixDisplay.from_predictions(
        y_true=y_test,
        y_pred=y_pred,
        display_labels=labels,
        xticks_rotation='vertical',
        cmap=plt.cm.Blues
    )
    display.ax_.set_title('Plot of the Confusion Matrix')
    plt.grid(False)
    plt.show()

    if y_proba is not None:
        print('ROC AUC: {:.3f}'.format(roc_auc_score(
            y_true=y_test, y_score=y_proba)))
        display = PrecisionRecallDisplay.from_predictions(
            y_true=y_test, y_pred=y_proba)
        display.ax_.set_title('Precision-Recall curve for test labels')
        plt.grid(True)
        plt.show()


def display_roc_curve(y_test, y_proba):
    """This function plot the ROC curve.

    Args:
        y_test (array-like): the test labels
        y_proba (array-like): the predicted probabilities
    """

    # Plot the ROC curve
    plt.style.use('seaborn')
    fpr, tpr, thr = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fpr, tpr, color='red', label='AUC={:.2f}'.format(auc(fpr, tpr)))
    ax.legend(loc='lower right')
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.set_title('Receiver Operating Characteristic')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.axis('tight')
    ax.grid(True)
    plt.show()
