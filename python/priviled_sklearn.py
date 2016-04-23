from utils import softmax
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.linear_model import LassoLarsCV, LassoLarsIC
from paths import stats_pca_path, cord_length_path, stats_files

teachers = [RidgeCV(), LassoCV(), ElasticNetCV(), LassoLarsIC(), LassoLarsCV()]

teacher_names = ['Ridge', 'LassoCV', 'Elastic Net',
                 'LassoLarsIC', 'LassoLarsCV']

print 'Stats PCA', stats_pca_path()
print 'Cordlength', cord_length_path()
print 'Stats files', stats_files()


