import numpy as np

# Custom implementation of K-means clustering
class KMeans:
    def __init__(self, n_clusters, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X):
        # Set random seed for reproducibility, if specified
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Initialize cluster centers randomly
        self.cluster_centers = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        # Repeat clustering process for a fixed number of iterations
        for i in range(self.max_iter):
            # Assign each point to its nearest cluster center
            distances = np.linalg.norm(X[:, np.newaxis, :] - self.cluster_centers, axis=2)
            labels = np.argmin(distances, axis=1)

            # Update cluster centers as the mean of their assigned points
            for j in range(self.n_clusters):
                self.cluster_centers[j] = np.mean(X[labels == j], axis=0)

# Radial Basis Function Network (RBFN)
class RBFN:
    def __init__(self, n_hidden, sigma=1.0):
        self.n_hidden = n_hidden  # Number of radial basis functions
        self.sigma = sigma  # Width of Gaussian RBFs
        self.centers = None  # Centers of the radial basis functions
        self.weights = None  # Weights connecting RBFs to output layer

    # Gaussian radial basis function
    def _gaussian(self, X, center):
        return np.exp(-self.sigma*np.linalg.norm(X-center)**2)

    # Calculate the centers of the radial basis functions using K-means clustering
    def _calculate_centers(self, X):
        kmeans = KMeans(n_clusters=self.n_hidden, random_state=0)
        kmeans.fit(X)
        centers = kmeans.cluster_centers
        return centers

    # Calculate the weights connecting the radial basis functions to the output layer
    def _calculate_weights(self, X, y):
        # Compute the outputs of the radial basis functions for each input
        Z = np.zeros((X.shape[0], self.n_hidden))
        for i in range(X.shape[0]):
            for j in range(self.n_hidden):
                Z[i,j] = self._gaussian(X[i], self.centers[j])

        # Compute the weights using the pseudoinverse of the RBF outputs
        self.weights = np.dot(np.linalg.pinv(Z), y)

    # Fit the RBFN to the training data
    def fit(self, X, y):
        # Compute the centers of the radial basis functions
        self.centers = self._calculate_centers(X)

        # Compute the weights connecting the RBFs to the output layer
        self._calculate_weights(X, y)

    # Make predictions on new data
    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            # Compute the weighted sum of the RBF outputs for each input
            summation = 0
            for j in range(self.n_hidden):
                summation += self.weights[j] * self._gaussian(X[i], self.centers[j])

            # Round the output to the nearest integer (assuming binary classification)
            y_pred[i] = np.round(summation)

        return y_pred

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    # Generate some sample data
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Create RBFN object and fit to training data
    rbfn = RBFN(n_hidden=10, sigma=0.1)
    rbfn.fit(X_train, y_train)

    # Make predictions on testing data
    y_pred = rbfn.predict(X_test)

    # Compute accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: {:.2f}%".format(accuracy*100))
