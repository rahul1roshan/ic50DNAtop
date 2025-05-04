from .kernel import TanimotoKernel
from sklearn.svm._base import BaseSVC
import numpy as np
from scipy import sparse


class TanimotoKNN:
    def __init__(self, n_neighbors=1):
        self._params = dict()
        self._params["n_neighbors"] = n_neighbors
        self._training_feature_mat = None
        self._training_labels = None

    def fit(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise IndexError("Not same shape")
        self._training_feature_mat = X
        self._training_labels = y
        return self

    def predict(self, X):
        sim_mat = TanimotoKernel.similarity_from_sparse(X, self._training_feature_mat)
        k = self._params["n_neighbors"]
        # get k last indices (k instances with highest similarity) for each row
        nn_list = np.argsort(sim_mat, axis=1)[:, -k:]
        predicted = []
        for nns in nn_list:
            assert len(nns) == k
            nn_label = self._training_labels[nns]
            label, label_occ = np.unique(nn_label, return_counts=True)
            predicted.append(label[np.argmax(label_occ)])
        return np.array(predicted)

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)

    def get_params(self, deep=True):
        return self._params

    def set_params(self, **params):
        self._params = params
        return self


class ExplainingSVC(BaseSVC):
    """ SVC copied form sklearn and modified

    """
    _impl = 'c_svc'

    def __init__(self, *, C=1.0, degree=3, gamma='scale',
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=200, class_weight=None,
                 verbose=False, max_iter=-1, decision_function_shape='ovr',
                 break_ties=False,
                 random_state=None):
        super().__init__(
            kernel=TanimotoKernel.similarity_from_sparse, degree=degree, gamma=gamma,
            coef0=coef0, tol=tol, C=C, nu=0., shrinking=shrinking,
            probability=probability, cache_size=cache_size,
            class_weight=class_weight, verbose=verbose, max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state)
        self._explicit_support_vectors = None

    def fit(self, X, y, sample_weight=None):
        x = super().fit(X, y, sample_weight=sample_weight)
        idx = self.support_
        self._explicit_support_vectors = X[idx]
        return self

    def vector_feature_weights(self, vector):
        suport_vectors = self.explicid_support_vectors
        dual_coeff = self.dual_coef_
        if dual_coeff.shape[0] > 1:
            raise NotImplementedError("Only binary Models are supported")
        dual_coeff = dual_coeff.reshape(-1, 1)

        norm_1 = np.array(vector.multiply(vector).sum(axis=1))
        norm_2 = np.array(suport_vectors.multiply(suport_vectors).sum(axis=1))
        prod = vector.dot(suport_vectors.T).toarray()
        denominator = 1 / (norm_1 + norm_2.T - prod)

        shared_features = vector.multiply(suport_vectors)
        upper_term = shared_features.multiply(dual_coeff).toarray()
        fw = denominator.dot(upper_term)
        return fw

    @property
    def explicid_support_vectors(self):
        return self._explicit_support_vectors

    def feature_weights(self, x: sparse.csr_matrix):
        return np.vstack([self.vector_feature_weights(x[i, :]) for i in range(x.shape[0])])
