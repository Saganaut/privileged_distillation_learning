from keras.models import Sequential, Graph
from keras.objectives import categorical_crossentropy 
from keras.layers.core import Dense, Dropout, Activation
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import theano.tensor as T
import numpy as np
import theano
from utils import load_chords, get_chords_pca, get_metadata
from paths import stats_pca_path, cord_length_path, stats_files

# experiment 1: noiseless labels as privileged info
def load_data(num_chord_comp=5, num_grain_comp=5):
    # Grain data
    grain_pca = np.load(stats_pca_path()+'grain_grain_pca_scores.npy')
    #load chord data
    chords = load_chords(cord_length_path())
    chords_pca = get_chords_pca(chords, use_avg=True) 
    #load labels
    metadata, class_map, subclass_map = get_metadata(stats_files())
    classes = np.array([int(x['class_num']) for x in metadata])
    #subclasses = np.array([x['subclass_num'] for x in metadata])
    #         xs          x         y
    return (grain_pca[:, :num_grain_comp] ,chords_pca[:, :num_chord_comp], classes)

def MLP(d,q):
    model = Sequential()
    model.add(Dense(q, input_dim=d))
    model.add(Activation('softmax'))
    model.compile('rmsprop','categorical_crossentropy')
    return model

def softmax(w, t = 1.0):
    e = np.exp(w / t)
    return e/np.sum(e,1)[:,np.newaxis]

def weighted_loss(base_loss,l):
    def loss_function(y_true, y_pred):
        return l*base_loss(y_true,y_pred)
    return loss_function

def distillation(d,q,t,l):
    graph = Graph()
    graph.add_input(name='x', input_shape=(d,))
    graph.add_node(Dense(q), name='w3', input='x')
    graph.add_node(Activation('softmax'), name='hard_softmax', input='w3')
    graph.add_node(Activation('softmax'), name='soft_softmax', input='w3')
    graph.add_output(name='hard', input='hard_softmax')
    graph.add_output(name='soft', input='soft_softmax')

    loss_hard = weighted_loss(categorical_crossentropy,1.-l)
    loss_soft = weighted_loss(categorical_crossentropy,t*t*l)

    graph.compile('rmsprop', {'hard':loss_hard, 'soft':loss_soft})
    return graph

def do_exp(x_tr,xs_tr,y_tr,x_te,xs_te,y_te):
    t = 1
    l = 1
    # scale stuff
    s_x   = StandardScaler().fit(x_tr)
    s_s   = StandardScaler().fit(xs_tr)
    x_tr  = s_x.transform(x_tr)
    x_te  = s_x.transform(x_te)
    xs_tr = s_s.transform(xs_tr)
    xs_te = s_s.transform(xs_te)
    y_tr  = y_tr*1.0
    y_te  = y_te*1.0
    y_tr  = np.vstack((y_tr==1,y_tr==0)).T
    y_te  = np.vstack((y_te==1,y_te==0)).T
    # privileged baseline
    mlp_priv = MLP(xs_tr.shape[1],2)
    mlp_priv.fit(xs_tr, y_tr, nb_epoch=1000, verbose=0)
    res_priv = np.mean(mlp_priv.predict_classes(xs_te,verbose=0)==np.argmax(y_te,1))
    # unprivivileged baseline
    mlp_reg = MLP(x_tr.shape[1],2)
    mlp_reg.fit(x_tr, y_tr, nb_epoch=1000, verbose=0)
    res_reg = np.mean(mlp_reg.predict_classes(x_te,verbose=0)==np.argmax(y_te,1))
    # distilled
    mlp_dist = distillation(x_tr.shape[1],2,t,l)
    #print type(mlp_priv.layers[0].call(xs_tr))
    soften = theano.function([mlp_priv.layers[0].input], mlp_priv.layers[0].call(xs_tr),  on_unused_input='ignore')#get_output(train=False))
    p_tr   = softmax(soften(xs_tr.astype(np.float32)),t)
    mlp_dist.fit({'x':x_tr, 'hard':y_tr, 'soft':p_tr}, nb_epoch=1000, verbose=0)
    res_dis = np.mean(np.argmax(mlp_dist.predict({'x':x_te},verbose=0)['hard'],1)==np.argmax(y_te,1))
    return np.array([res_priv,res_reg,res_dis])


def split_data(xs, x, y):
    #Stack the two so I can split using same index, lazy style
    combo = np.hstack((xs,x))
    combo_tr, combo_te, y_tr, y_te = train_test_split(combo, y, test_size = .2)
    xs_tr = combo_tr[:, :xs.shape[1]]
    xs_te = combo_te[:, :xs.shape[1]]
    x_tr = combo_tr[:, xs.shape[1]:]
    x_te = combo_te[:, xs.shape[1]:]
    return xs_tr, x_tr, y_tr, xs_te, x_te, y_te


if __name__ == '__main__':
    # experiment hyper-parameters
    d      = 50
    n_tr   = 200
    n_te   = 1000
    n_reps = 5

    np.random.seed(0)
    performance_mat = np.zeros((n_reps,3))
    xs, x, y = load_data()
    for rep in xrange(n_reps):
        print '~~~~~~ Running Trial: ' + str(rep)
        a = np.random.randn(d)
        xs_tr, x_tr, y_tr, xs_te, x_te, y_te = split_data(xs, x, y) 
        performance_mat[rep,:] += do_exp(x_tr,xs_tr,y_tr,x_te,xs_te,y_te)
    means = performance_mat.mean(axis=0).round(4)
    stds  = performance_mat.std(axis=0).round(4)
    print 'Priv: '+str(means[0])+' +- '+str(stds[0])+\
          '\nNorm: '+str(means[1])+' +- '+str(stds[1])+\
          '\nDist: '+str(means[2])+' +- '+str(stds[2])

