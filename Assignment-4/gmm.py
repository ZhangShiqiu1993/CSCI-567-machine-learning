import numpy as np
from kmeans import KMeans


class GMM():
    """
        Fits a Gausian Mixture model to the data.

        attrs:
            n_cluster : Number of mixtures
            e : error tolerance
            max_iter : maximum number of updates
            init : initialization of means and variance
                Can be 'random' or 'kmeans'
            means : means of gaussian mixtures
            variances : variance of gaussian mixtures
            pi_k : mixture probabilities of different component
    """

    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None

    def fit(self, x):
        """
            Fits a GMM to x.

            x: is a NXD size numpy array
            updates:
                self.means
                self.variances
                self.pi_k
        """
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape

        if self.init == 'k_means':
            kmeans = KMeans(self.n_cluster, self.max_iter, self.e)
            means, membership, _ = kmeans.fit(x)
            self.means = means

            self.variances = np.zeros((self.n_cluster, D, D))
            for k in range(self.n_cluster):
                mask = membership == k
                x_k = x[mask] - self.means[k]
                self.variances[k] = np.dot(x_k.T, x_k) / np.sum(mask)
            self.pi_k = np.bincount(membership) / N
        elif self.init == 'random':
            self.means = np.array([np.random.uniform(0, 1, D) for _ in range(self.n_cluster)])
            self.variances = np.array([np.eye(D) for _ in range(self.n_cluster)])
            self.pi_k = np.ones(self.n_cluster) / self.n_cluster
        else:
            raise Exception('Invalid initialization provided')

        gamma = np.zeros((N, self.n_cluster))

        l = self.compute_log_likelihood(x)

        for i in range(self.max_iter):
            # E step
            for k in range(self.n_cluster):
                gamma[:, k] = self.pi_k[k] * self.pdf(x, self.means[k], self.variances[k])
            gamma = np.apply_along_axis(lambda r: r / np.sum(r), axis=1, arr=gamma)

            n_k = np.sum(gamma, axis=0)

            # M step
            for k in range(self.n_cluster):
                self.means[k] = np.sum(gamma[:, k] * x.T, axis=1).T / n_k[k]
                x_mu = np.matrix(x - self.means[k])
                self.variances[k] = np.dot(np.multiply(x_mu.T, gamma[:, k]), x_mu) / n_k[k]
                self.pi_k[k] = n_k[k] / N
            l_new = self.compute_log_likelihood(x)

            if np.abs(l - l_new) < self.e:
                break
            l = l_new
        return i

    def pdf(self, x, mean, variance):
        k = len(mean)
        while np.linalg.det(variance) == 0:
            variance += np.eye(variance.shape[0]) * 0.001
        inv_var = np.linalg.inv(variance)
        x_mu = x - mean
        numerator = np.exp(-0.5 * np.einsum('ij, ji -> i', x_mu, np.dot(inv_var, x_mu.T)))
        denominator = np.sqrt((2 * np.pi) ** k * np.linalg.det(variance))
        return numerator / denominator

    def sample(self, N):
        """
        sample from the GMM model

        N is a positive integer
        return : NXD array of samples

        """
        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        if self.means is None:
            raise Exception('Train GMM before sampling')

        D = self.means.shape[1]
        samples = np.zeros((N, D))
        choice = np.random.choice(self.n_cluster, N, p=self.pi_k)
        for i in range(N):
            mean = self.means[choice[i]]
            var = self.variances[choice[i]]
            samples[i] = np.random.multivariate_normal(mean, var)
        return samples

    def compute_log_likelihood(self, x):
        """
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        """
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        gamma = np.zeros(x.shape[0])
        for k in range(self.n_cluster):
            gamma += self.pi_k[k] * self.pdf(x, self.means[k], self.variances[k])
        score = np.sum(np.log(gamma))
        return float(score)
