import math
import numpy as np
from decisionTree import DecisionTreeClassifier
from scipy import stats

# Based on Decision Tree class
# Uses multiple "weaker" trees to determine optimal predicion
class RandomForest:
    def __init__(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        self.forest = []

    def bootstrapAggregation(self, x):
        # Get indicies of the random set of features

        # featureX is the number of features to use in the new tree
        feature_X = math.ceil(math.sqrt(len(x[0])))

        # Pick a random set of features without replacement and return indexes to an array
        random_features = np.random.choice(len(x[0]), feature_X, replace=False)

        # Pick a random set of rows with replacement and return indexes to an array
        bootstrapped_data = np.random.choice(len(x), len(x), replace=True)

        # Return both feature and data indexes
        return random_features, bootstrapped_data

    def randomization(self, x, y):

        # Get values of randomized data and create a new dataset with them
        random_features, bootstrapped_data = self.bootstrapAggregation(x)

        # Create new set of rows and output columns
        random_subset = x[bootstrapped_data][:, random_features]
        random_y = y[bootstrapped_data]

        # Return randomized data, output columns, and orginal feature indexes for correction
        return random_subset, random_y, random_features

    # Max depth should be about 20
    def randomForest(self, n, max_depth):

        # Accumulate n random sets of data
        data = [self.randomization(self.X_train, self.Y_train) for _ in range(n)]

        # Create an array of different Decision Trees of a preset max depth
        forest = [DecisionTreeClassifier(max_depth=max_depth) for _ in range(len(data))]

        # Fit each tree with the respective data
        for tree, (x, y, z) in enumerate(data):
            forest[tree].fit(x, y, z)

        # Update forest instance variable
        self.forest = forest

    def predict(self, x):

        # Return mode of all the predictions from each tree
        # This predicts the class that the item is a part of
        return stats.mode([t.predict(x) for t in self.forest], axis=0)[0].reshape(-1)
