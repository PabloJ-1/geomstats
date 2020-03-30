"""
Grassmannian manifold Gr(n, k),
a set of all k-dimensional subspaces in n-dimensional space,
where k <= n
"""

import geomstats.backend as gs
from geomstats.geometry.embedded_manifold import EmbeddedManifold
from geomstats.geometry.matrices_space import MatricesMetric
from geomstats.geometry.matrices_space import MatricesSpace
from geomstats.geometry.riemannian_metric import RiemannianMetric

TOLERANCE = 1e-5
EPSILON = 1e-6


class Grassmannian(EmbeddedManifold):
    """
    Class for Grassmann manifolds Gr(n, k)
    of k-dimensional subspaces in the n-dimensional euclidean space.
        Parameters
        ----------
        point : array-like, shape=[n_samples, n, k]
            Point.
        tolerance : float, optional
            Tolerance at which to evaluate.

        Returns
        -------
        belongs : array-like, shape=[n_samples, 1]
            Array of booleans evaluating if the corresponding points
            belong to the Grassmann manifold.
    """
    def __init__(self, n, k):
        assert isinstance(n, int) and isinstance(k, int)
        assert k <= n

        self.n = n
        self.k = k

        dimension = int(k * (n - k))
        super(Grassmannian, self).__init__(
              dimension=dimension,
              embedding_manifold=MatricesSpace(n, n))

    def belongs(self, point, tolerance=TOLERANCE):
        """
        Check if an (n,n)-matrix is an orthogonal projector
        onto a subspace of rank k.
        """
        point = gs.to_ndarray(point, to_ndim=3)
        n_points, n, k = point.shape

        if (n, k) != (self.n, self.k):
            return gs.array([[False]] * n_points)

        point_transpose = gs.transpose(point, axes=(0, 2, 1))
        identity = gs.to_ndarray(gs.eye(k), to_ndim=3)
        identity = gs.tile(identity, (n_points, 1, 1))
        diff = gs.einsum('nij,njk->nik', point_transpose, point) - identity

        diff_norm = gs.linalg.norm(diff, axis=(1, 2))
        belongs = gs.less_equal(diff_norm, tolerance)

        belongs = gs.to_ndarray(belongs, to_ndim=1)
        belongs = gs.to_ndarray(belongs, to_ndim=2, axis=1)
        return belongs
        
        
        raise NotImplementedError(
                'The Grassmann `belongs` is not implemented.'
                'It shall test whether p*=p, p^2 = p and rank(p) = k.')

    def origin(self):
        return gs.diag(gs.repeat([1, 0], [self.k, self.n - self.k]))[0]


class GrassmannianCanonicalMetric(RiemannianMetric):
    """
    Canonical metric of the Grassmann manifold.

    Coincides with the Frobenius metric.
    """
    def __init__(self, n, k):
        assert isinstance(n, int) and isinstance(k, int)
        assert k <= n
        self.n = n
        self.k = k
        dimension = int(k * (n - k))

        super(GrassmannianCanonicalMetric, self).__init__(
                dimension=dimension,
                signature=(dimension, 0, 0))

        self.embedding_metric = MatricesMetric(n, n)

        self.manifold = Grassmannian(n, k)

    def exp(self, vector, point):
        """
        Exponentiate the invariant vector field v from base point p.

        `vector` is skew-symmetric, in so(n).
        `point` is a rank p projector of Gr(n, k).

        Parameters
        ----------
        vector : array-like, shape=[n_samples, n, n]
        point : array-like, shape=[n_samples, n, n]

        Returns
        -------
        exp : array-like, shape=[n_samples, n, n]
        """
        expm = gs.linalg.expm
        mul = MatricesSpace.mul
        return mul(mul(expm(vector), point), expm(-vector))

    def log(self, point, base_point):
        """
        Riemannian logarithm of a point from a base point.

        Returns a skew-symmetric matrix in the image of [so(n), point].

        Parameters:
        ----------
        point : array-like, shape=[n_samples, n, n]
        base_point : array-like, shape=[n_samples, n, n]

        Returns:
        -------
        log : array-like, shape=[n_samples, n, n]
        """
        svd = gs.linalg.svd
        mul = gs.matmul
        tr = MatricesSpace.transpose
        logm = gs.linalg.logm

        def closest(rot):
            d_coefs = gs.diagonal(rot)
            d_sign = gs.where(d_coefs >= 0, 1., -1.)
            return mul(rot, gs.diag(d_sign)[0])

        rot2 = svd(point)[0]
        rot1 = svd(base_point)[0]
        rot = mul(rot2, tr(rot1))

        return logm(closest(rot))