import numpy as np
import os
from sklearn.decomposition import PCA


def softmax(X, copy=True):
    """
    Calculate the softmax function.

    The softmax function is calculated by
    np.exp(X) / np.sum(np.exp(X), axis=1)

    This will cause overflow when large values are exponentiated.
    Hence the largest value in each row is subtracted from each data
    point to prevent this.

    Parameters
    ----------
    X: array-like, shape (M, N)
        Argument to the logistic function

    copy: bool, optional
        Copy X or not.

    Returns
    -------
    out: array, shape (M, N)
        Softmax function evaluated at every point in x
    """
    if copy:
        X = np.copy(X)
    max_prob = np.max(X, axis=1).reshape((-1, 1))
    X -= max_prob
    np.exp(X, X)
    sum_prob = np.sum(X, axis=1).reshape((-1, 1))
    X /= sum_prob
    return X


def get_filtered_count(files, filter_phrase):
    n = 0
    for fi in files:
        if fi[-3:] != 'npy':
            continue
        tokens = fi.split('.')
        if not (filter_phrase in tokens):
            continue
        n += 1
    return n


def load_data(stats_files):

    files = sorted(os.listdir(stats_files))
    c = 0
    while (files[c][-3:] != 'npy'):
        c += 1
    x = np.zeros((get_filtered_count(files, '1_stats'),
                  np.prod(np.load(stats_files+files[c]).shape)))
    n = 0
    for fi in files:
        if fi[-3:] != 'npy':
            continue
        tokens = fi.split('.')
        if '1_stats' not in tokens:
            continue
        full = os.path.join(stats_files, fi)
        print '->Processed ' + str(n) + ' ' + fi
        temp = np.load(full)
        x[n, :] = temp.flatten()
        n += 1
    return x


def get_metadata(path_root):
    files = os.listdir(path_root)
    metadata = []
    class_map = {}
    c = 0
    for fi in files:
        if fi[-3:] != 'npy':
            continue
        temp = {}
        tokens = fi.split('.')
        if '1_stats' not in tokens:
            continue
        temp['class'] = tokens[0]
        temp['subclass'] = tokens[1]
        temp['volume_frac'] = tokens[4]
        if not temp['class'] in class_map.keys():
            class_map[temp['class']] = c
            c += 1
        temp['class_num'] = class_map[temp['class']]
        metadata.append(temp)
    print 'Classes' + str(class_map)
    return metadata, class_map


def get_pca(x, num_comp=10):
    print '->Performing PCA'
    pca = PCA(n_components=num_comp)
    x_pca = pca.fit_transform(x)
    #  draw_component_variance(pca.explained_variance_ratio_)
    return x_pca
