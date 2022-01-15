import subprocess

from KNN import KNNClassifier
from utils import *

target_attribute = 'Outcome'


def run_knn(k, x_train, y_train, x_test, y_test, formatted_print=True):
    neigh = KNNClassifier(k=k)
    neigh.train(x_train, y_train)
    y_pred = neigh.predict(x_test)
    acc = accuracy(y_test, y_pred)
    print(f'{acc * 100:.2f}%' if formatted_print else acc)


def get_top_b_features(x, y, b=5, k=51):
    """
    :param k: Number of nearest neighbors.
    :param x: array-like of shape (n_samples, n_features).
    :param y: array-like of shape (n_samples,).
    :param b: number of features to be selected.
    :param method: sklearn method
    :return: indices of top 'b' features as the result of selection/dimensionality reduction on sample
            sets using sklearn.feature_selection module
    """
    # TODO: Implement get_top_b_features function
    #   - Note: The brute force approach which examines all subsets of size `b` will not be accepted.

    assert 0 < b < x.shape[1], f'm should be 0 < b <= n_features = {x.shape[1]}; got b={b}.'
    top_b_features_indices = []

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # assume the features are {1, 2, 3, ..., d}
    # set features_group = {1, 2, 3, ..., d}
    # while len(features_group) > b:
    #   for each subset of features_group with size len(features_group)-1:
    #       run K-fold on this subset of features_group
    #       save the accuracy of K-fold run
    #   set features_group to the subset with the best accuracy
    # return features_group
    #
    # Example:
    # We start with features {1, 2, 3, 4}
    # We run K-fold on the sets:
    #   {1, 2, 3}
    #   {1, 2, 4}
    #   {1, 3, 4}
    #   {2, 3, 4}
    #
    # Now we choose the set with the best accuracy.
    # Lets say it's the set {1, 2, 4}
    # Now we run K-fold on the sets:
    # {1, 2}
    # {1, 4}
    # {2, 4}
    #
    # Now choose the set with the best accuracy
    # And so on... until we get a set with size == b and return it
    # ========================

    return top_b_features_indices


def run_cross_validation():
    """
    cross validation experiment, k_choices = [1, 5, 11, 21, 31, 51, 131, 201]
    """
    file_path = str(pathlib.Path(__file__).parent.absolute().joinpath("KNN_CV.pyc"))
    subprocess.run(['python', file_path])


def exp_print(to_print):
    print(to_print + ' ' * (30 - len(to_print)), end='')


# ========================================================================
if __name__ == '__main__':
    """
       Usages helper:
       (*) cross validation experiment
            To run the cross validation experiment over the K,Threshold hyper-parameters
            uncomment below code and run it
    """
    run_cross_validation()

    # # ========================================================================

    attributes_names, train_dataset, test_dataset = load_data_set('KNN')
    x_train, y_train, x_test, y_test = get_dataset_split(train_set=train_dataset,
                                                         test_set=test_dataset,
                                                         target_attribute='Outcome')

    best_k = 51
    b = 0

    # # ========================================================================

    print("-" * 10 + f'k  = {best_k}' + "-" * 10)
    exp_print('KNN in raw data: ')
    run_knn(best_k, x_train, y_train, x_test, y_test)

    top_m = get_top_b_features(x_train, y_train, b=b, k=best_k)
    x_train_new = x_train[:, top_m]
    x_test_test = x_test[:, top_m]
    exp_print(f'KNN in selected feature data: ')
    run_knn(best_k, x_train_new, y_train, x_test_test, y_test)
