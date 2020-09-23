import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

def visualize_evaluation(matrix, index_names = None, column_names = None, x_label = None, y_label = None, title = None):
    _, axes = plt.subplots(1, 2, figsize = (12, 5))

    sn.set(font_scale = 0.8)
    for idx in range(len(matrix)):
        df_matrix = pd.DataFrame(matrix[idx], index = index_names[idx], columns = column_names[idx])
        sn.heatmap(df_matrix, annot = True, ax = axes[idx])
        axes[idx].set_title(title[idx])
        axes[idx].set_xlabel(x_label[idx])
        axes[idx].set_ylabel(y_label[idx])
    plt.tight_layout()
    plt.show()

def model_evaluate(model, test_data):
    """
    Precision - Recall, F1-Score for multi-class classification sử dụng confusion matrix
    Default model: f(data) = label
    Default test_data = numpy array([(data, label), (data, label),...])
    visualize confusion matrix and model evaluation using matplotlib, seaborn
    return tuple(accuracy, ndarray([precision, recall, F1-Score], [precision, recall, F1-Score],...)
    """

    # prepare data and label
    labels = np.array([(model(data), label_true) for data, label_true in test_data])
    label_pred = labels[:, 0]
    label_true = labels[:, 1]
    class_names = np.unique(label_true)
    num_classes = class_names.shape[0]
    
    # confusion matrix
    cfs_matrix = np.zeros((num_classes, num_classes))
    for idx in range(label_true.shape[0]):
        cfs_matrix[label_true[idx], label_pred[idx]] += 1
    
    try:
        # compare with confusion matrix when using sklearn libary
        from sklearn.metrics import confusion_matrix
        cfs_matrix_sklearn = confusion_matrix(label_true, label_pred)

        if cfs_matrix.all() != cfs_matrix_sklearn.all():
            raise Exception
    except Exception:
        print('Something is wrong with confusion matrix')
    else:
        # calculate accuracy
        accuracy = float(np.diagonal(cfs_matrix).sum())/cfs_matrix.sum()
        
        # precision - recall, F1-score evaluate
        result_evaluate = np.empty((3, num_classes))
        for field in class_names:
            precision = float(cfs_matrix[field, field])/(cfs_matrix[:, field].sum())
            recall = float(cfs_matrix[field, field])/(cfs_matrix[field, :].sum())
            f1_score = 2 * (float(precision * recall)/ (precision + recall))
            result_evaluate[:, field] = np.array([precision, recall, f1_score])

        # prepare and visuzalize confusion matrix and model valuation
        matrix_init = np.concatenate((np.array([cfs_matrix]), np.array([result_evaluate])), axis = 0)
        index_names_init = np.concatenate((np.array([class_names]), np.array([['precision', 'recall', 'F1-Score']])), axis = 0)
        column_names_init = np.concatenate((np.array([class_names]), np.array([class_names])), axis = 0)
        xlabel_init = np.concatenate((['Predicted Label'], [None]), axis = 0)
        ylabel_init = np.concatenate((['True Label'], [None]), axis = 0)
        title_init = np.concatenate((['Unnormalized Confusion Matrix'], ['Model Evaluation']), axis = 0)

        visualize_evaluation(matrix_init, index_names_init, column_names_init, xlabel_init, ylabel_init, title_init)

        return (accuracy, result_evaluate.T)
    finally:
        print('Done evaluate')


