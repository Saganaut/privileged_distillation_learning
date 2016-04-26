import numpy as np
import os
import sys
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import colors
import paths

from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model, cross_validation
from utils import load_data, get_pca, get_metadata, load_chords, plot_chords, group_components_by_class, get_chords_pca
from pymks.tools import draw_component_variance, draw_components
from sklearn.tree import DecisionTreeRegressor


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

def flatten_input_params(metadata, class_map, subclass_map):
    #Flatten down inputs into binary vars
    total_classes = set()
    total_subclasses = set()
    for item in metadata:
        total_classes.add(item['class'])
        total_subclasses.add(item['subclass'])

    y = np.zeros((len(metadata), len(total_classes)+len(total_subclasses) + 1))
    c = 0  
    # y = [ class binary vars | subclass binary vars | pct] 
    for item in metadata:
        y[c, class_map[item['class']]] = 1
        y[c, subclass_map[item['subclass']] + len(total_classes) ] = 1
        #TODO fix the filenames or write a script to handle vol frac
#        print item['volume_frac']
#        y[c, len(total_classes) + len(total_subclasses)] = item['volume_frac']
        c += 1 
    return y

if __name__ == '__main__':
    num_grain_comp = 15
    num_chord_comp = 3
    num_folds = 5

    metadata, class_map, subclass_map = get_metadata(paths.stats_files())
    if os.path.isfile(paths.stats_pca_path()+'grain_grain_pca_scores.npy'):
        print 'PCA .npy found, loading.'
        pca_scores = np.load(paths.stats_pca_path()+'grain_grain_pca_scores.npy')
    else:
        x = load_data(paths.stats_files())
        pca_scores = get_pca(x)
        np.save(paths.stats_pca_path()+'grain_grain_pca_scores.npy', pca_scores)
    chords = load_chords(paths.cord_length_path())
    chords_pca = get_chords_pca(chords, use_avg=True) 
    input_params = flatten_input_params(metadata, class_map, subclass_map)    
      

    # PLOTTING FCNS
#    plot_chords(chords[0:5,0])
#    class_labels, class_data = group_components_by_class(metadata, chords_pca)
#    draw_components(class_data, class_labels)
    
  
    linreg = linear_model.LinearRegression(normalize=False, fit_intercept=True)
    scores = cross_validation.cross_val_score(linreg,
                                              input_params,
                                              chords_pca[:, :num_chord_comp],
                                              cv=num_folds)
    print 'LinReg: ' + str(np.mean(scores))

    alphas = np.linspace(.001, .8, 10)
    for alpha in alphas:
        ridge = linear_model.Ridge(alpha=alpha)
        scores = cross_validation.cross_val_score(ridge, 
                                                  input_params, 
                                                  chords_pca[:, :num_chord_comp], 
                                                  cv=num_folds)
        print 'Ridge (alpha='+str(alpha)+'): ' + str(np.mean(scores))
    lasso = linear_model.Lasso(alpha=0.5)
    scores = cross_validation.cross_val_score(lasso, 
                                              input_params, 
                                              chords_pca[:, :num_chord_comp], 
                                              cv=num_folds)
    print 'Lasso: ' + str(np.mean(scores))
    regressor = DecisionTreeRegressor(random_state=0)
    scores = cross_validation.cross_val_score(regressor, 
                                              input_params, 
                                              chords_pca[:,:num_chord_comp], 
                                              cv=num_folds)
    print 'DT: ' + str(np.mean(scores))

    #~~~~~~~~~~~~~~~~~~~Do grain linkages

    grain_pca = pca_scores[:, 0:num_grain_comp]
    ridge = linear_model.Ridge(alpha=.01)
    scores = cross_validation.cross_val_score(ridge, 
                                              grain_pca, 
                                              chords_pca[:, :num_chord_comp], 
                                              cv=num_folds)
    print 'Ridge (alpha='+str(alpha)+'): ' + str(np.mean(scores))
    lasso = linear_model.Lasso(alpha=0.01)
    scores = cross_validation.cross_val_score(lasso, 
                                              grain_pca, 
                                              chords_pca[:, :num_chord_comp], 
                                              cv=num_folds)
    print 'Lasso: ' + str(np.mean(scores))
    regressor = DecisionTreeRegressor(random_state=0)
    scores = cross_validation.cross_val_score(regressor, 
                                              grain_pca[:, :num_chord_comp], 
                                              chords_pca, 
                                              cv=num_folds)
    print 'DT: ' + str(np.mean(scores))

