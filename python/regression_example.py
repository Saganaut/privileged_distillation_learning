from priviledged_regression import PriviledgedLinearRegression
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.cross_validation import train_test_split


teacher_model = LinearRegression()
student_model = LinearRegression()
distill_model = PriviledgedLinearRegression()


n_samples = 300
n_features = 1

np.random.seed(1)
y = 36. * np.random.random(n_samples)[:, None]
X_priv = y + 15. * np.random.random((len(y), n_features)) - 7.5
X = X_priv + 50. * np.random.random((len(y), n_features)) - 25


X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
    X_priv, y, test_size=0.2, random_state=42)

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X, y, test_size=0.2, random_state=42)


teacher_model.fit(X_train_t, y_train_t)
student_model.fit(X_train_s, y_train_s)
distill_model.fit(X_train_s, X_train_t, y_train_s)
print 'teacher', teacher_model.score(X_test_t, y_test_t)
print 'student', student_model.score(X_test_s, y_test_s)
print 'distillation', distill_model.score(X_test_s, y_test_s)
