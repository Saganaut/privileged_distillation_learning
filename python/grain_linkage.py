import numpy as np
import os
import sys
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import colors

from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from utils import load_data, get_pca, get_metadata, load_chords, get_chords_pca
from pymks.tools import draw_component_variance
from sklearn.cross_validation import cross_val_score
from paths import stats_pca_path, cord_length_path, stats_files

def run_classifiers(x, y, num_folds=5):
    i = 1
    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
             "Random Forest", "AdaBoost", "Naive Bayes",
             "Linear Discriminant Analysis", "Quadratic Discriminant Analysis"]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        AdaBoostClassifier(),
        GaussianNB(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis() ]

    for name, clf in zip(names, classifiers):
        scores = cross_val_score(clf, x, y, cv=num_folds)
        print name + ': ' + str(np.mean(scores))


if __name__ == '__main__':
    num_grain_comp = 5
    num_chord_comp = 5
    num_folds = 5

    metadata, class_map, subclass_map = get_metadata(stats_files())
    if os.path.isfile(stats_pca_path()+'grain_grain_pca_scores.npy'):
        print 'PCA .npy found, loading.'
        grain_pca = np.load(stats_pca_path()+'grain_grain_pca_scores.npy')
    else:
        x = load_data(stats_files())
        grain_pca = get_pca(x)
        np.save(stats_pca_path()+'grain_grain_pca_scores.npy', grain_pca)

    chords = load_chords(cord_length_path())
    chords_pca = get_chords_pca(chords, use_avg=True) 
    
    # show the pca plots
    #plt.scatter(grain_pca[:, 0], grain_pca[:,1], alpha=0.85)
    #plt.show()

    input_params = np.array([x['class_num'] for x in metadata])
   
    print '\n~~~~~~~~~~~~ Chord(PCA)->Class ~~~~~~~~~~~~'
    run_classifiers(chords_pca[:, :num_chord_comp], input_params)

    print '\n~~~~~~~~~~~~ Grain(PCA)->Class ~~~~~~~~~~~~'
    run_classifiers(grain_pca[:, :num_grain_comp], input_params)
                    
    quit()


if __name__ == '__main__':
    path_root = '/home/david/Research/grain_to_grain/truncated/'
    pca_root = '/home/david/Research/grain_to_grain/pca/'
    metadata, class_map = get_metadata(path_root)
    if os.path.isfile(pca_root+'grain_grain_pca_scores.npy'):
        print 'PCA .npy found, loading.'
        pca_scores = np.load(pca_root+'grain_grain_pca_scores.npy')
    else:
        x = load_data(path_root)
        pca_scores = get_pca(x)
        np.save(pca_root+'grain_grain_pca_scores.npy', pca_scores)

    # show the pca plots
    #plt.scatter(pca_scores[:, 0], pca_scores[:,1], alpha=0.85)
    #plt.show()

    mesh_height = 0.02
    i = 1
    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
             "Random Forest", "AdaBoost", "Naive Bayes",
             "Linear Discriminant Analysis", "Quadratic Discriminant Analysis"]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        AdaBoostClassifier(),
        GaussianNB(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis()]

    figure = plt.figure()

    color_map = colors.ListedColormap(['#1f77b4', '#ff7f0e', '#2ca02c',
                                       '#d62728', '#9467bd', '#8c564b',
                                       '#e377c2'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = colors.BoundaryNorm(bounds, color_map.N)
    # preprocess dataset, split into training and test part
    X = grain_pca[:, :5]
    print X.shape
    y = np.array([x['class_num'] for x in metadata])
    print y.shape
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_height),
                         np.arange(y_min, y_max, mesh_height))
    print xx.shape
    ax = plt.subplot(2, 5, i)
    # Plot the training points
    print y_train
    ax.scatter(X_train[:, 0], X_train[:, 1],
               c=y_train, cmap=color_map, norm=norm)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test,
               cmap=color_map, norm=norm, alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(2, 5, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        #if hasattr(clf, "decision_function"):
        #    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        #else:
        #    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        #print xx.shape
        #print Z.shape
        #sys.stdout.flush()
        #if np.prod(Z.shape) != np.prod(xx.shape): continue
        #Z = Z.reshape(xx.shape)
        #ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot also the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
                   cmap=color_map, norm=norm)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test,
                   cmap=color_map, norm=norm,
                   alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(name)
        ax.text(xx.max()/2.0 - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1
    figure.subplots_adjust(left=.02, right=.98)
    plt.show()
