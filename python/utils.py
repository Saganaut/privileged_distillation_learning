import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio 
from collections import defaultdict as dd
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


def load_chords(chords_files):
    files = sorted(os.listdir(chords_files))
    c = 0
    while (files[c][-3:] != 'mat'):
        c += 1
    chords = []
    for fi in files:
        if fi[-3:] != 'mat':
            continue
        if fi[-5:-4] != '1': continue
        full = os.path.join(chords_files, fi)
        a = sio.loadmat(full)
        chords.append([a['nhistogramavg'], 
                                a['nhistogramx'], 
                                a['nhistogramy'], 
                                a['nhistogramz']])
    return np.array(chords)


def get_metadata(path_root):
    files = os.listdir(path_root)
    metadata = []
    class_map = {}

    subclass_map = {}
    c = 0
    d = 0
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
        if not temp['subclass'] in subclass_map.keys():
            subclass_map[temp['subclass']] = d 
            d += 1
        temp['class_num'] = class_map[temp['class']]
        temp['subclass_num'] = subclass_map[temp['subclass']]
        metadata.append(temp)
    print 'Classes' + str(class_map)
    print 'Subclasses' + str(subclass_map)
    return metadata, class_map, subclass_map


def get_pca(x, num_comp=20):
    print '->Performing PCA'
    pca = PCA(n_components=num_comp)
    x_pca = pca.fit_transform(x)
    #  draw_component_variance(pca.explained_variance_ratio_)
    return x_pca

def get_chords_pca(chords, use_avg=False):
    print 'Getting PCA for chords'
    if use_avg:
        flat_chords= np.zeros((len(chords), chords[0].shape[1]))
    else:
        flat_chords = np.zeros((len(chords), 3*chords[0].shape[1]))
    c = 0
    for chord in chords:
        if use_avg:
            mega_chord = chord[0,:]
        else:
            mega_chord = np.vstack((chord[1,:], chord[2,:], chord[3,:]))
        flat_chords[c, :] = mega_chord.T
        c+=1 
    pca = PCA(n_components = 15)
    chords_pca = pca.fit_transform(flat_chords)
    #draw_component_variance(pca.explained_variance_ratio_)
    return chords_pca

def plot_chords(X):
    fig, ax = plt.subplots()
    ax.plot(np.tile(range(X.shape[1]), (X.shape[0], 1)).T, X[:,:,0].T, alpha=0.8)
    plt.show()
    

def group_components_by_class(metadata, components, truncate_dim=2):
    class_map = dd(list)
    c = 0
    for item in metadata:
        class_map[item['class']].append(components[c, :truncate_dim])
        c += 1 
    classes = []
    data = []
    for key in class_map.keys():
        classes.append(key)
        data.append(np.asarray(class_map[key]))
    return classes, data

