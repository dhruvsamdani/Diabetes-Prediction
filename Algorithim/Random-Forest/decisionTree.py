# Code taken from
# https://towardsdatascience.com/decision-tree-from-scratch-in-python-46e99dfea775
# There have been slight modifications the code from the website
# to better fit the needs of this project

import numpy as np

# Class is the final outcome of item we are trying to predict
# Feature is a characteristic of the item
class Node:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None


class DecisionTreeClassifier:

    # Max depth is how deep the tree will be
    def __init__(self, max_depth=None):
        self.tree = None
        self.number_features = None
        self.adjusted_feature_index = None
        self.number_classes = None
        self.max_depth = max_depth

    def getBestSplit(self, x, y):
        total_class_number = y.size
        if total_class_number < 2:
            return None, None

        # Gets the count of unique classes in the num classes data
        numClasses = [np.sum(y == c) for c in range(self.number_classes)]

        # Finds gini impurity for the current node
        currGini = 1 - sum((n / total_class_number) ** 2 for n in numClasses)

        bestFeature, best_threshold = None, None

        # Search through all the possible features
        for feature in range(self.number_features):

            # sort by features and then split back up into regular thresholds and class values
            # the thresholds values will be sorted; however, the classes will not as they are
            # the dependent variable to the thresholds
            thresholds, classes = zip(*sorted(zip(x[:, feature], y)))

            # Create counters for the left and right subtree
            num_left = [0] * self.number_classes
            num_right = numClasses.copy()

            # Loop through all possible class combinations
            for i in range(1, total_class_number):

                # Take value of class at ith index
                c = int(classes[i - 1])

                # Split up classes and calculate gini for both
                # num_right and num_left are the number of classes that will be in the right
                # and left subtree respectfully
                num_right[c] -= 1
                num_left[c] += 1

                # Calculate gini index for left, right and total for node
                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.number_classes))
                gini_right = 1.0 - sum((num_right[x] / (total_class_number - i)) ** 2 for x in range(self.number_classes))

                gini = (i * gini_left + (total_class_number - i) * gini_right) / total_class_number

                # If current threshold is the same as the previous one continue to find a more optial split
                # Since the thresholds are sorted we are allowed to do this to find the best split
                if thresholds[i] == thresholds[i - 1]:
                    continue

                # Compare gini indexes and update if necesary
                if gini < currGini:
                    currGini = gini
                    bestFeature = feature
                    best_threshold = (thresholds[i] + thresholds[i - 1]) / 2  # midpoint

        return bestFeature, best_threshold

    def fit(self, x, y, adjusted_feature=None):

        # Number of classes in the data set
        self.number_classes = len(set(y))
        # Number of features in the data set
        self.number_features = x.shape[1]

        # For random forest use
        # When random forest used, adjusted_feature_index will readjust feature index
        # to accomadate for the shift in index from the random forest
        self.adjusted_feature_index = adjusted_feature

        # grow tree
        self.tree = self.growTree(x, y)

    def growTree(self, x, y, depth=0):

        num_samples_per_class = [np.sum(y == i) for i in range(self.number_classes)]

        # Returns index of max element of num_samples_per_class
        predicted_class = np.argmax(num_samples_per_class)

        node = Node(predicted_class=predicted_class)

        # Check if max depth reached
        if depth < self.max_depth:

            # Get feature index and threshold from getBestSplit
            idx, thr = self.getBestSplit(x, y)

            if idx is not None:
                # Boolean arr for all the values that are under the threshold at
                # the specified index
                left = x[:, idx] < thr

                # Make left and right sub arrays
                # Left sub arrays are the values that have been filtered out i.e correct classes
                X_left, y_left = x[left], y[left]

                # Right sub arrays are the classes that have not been filtered out
                # They have not been through the threshold condition
                X_right, y_right = x[~left], y[~left]

                # Update information for the node
                # Check if adjusted index needs to be used
                node.feature_index = self.adjusted_feature_index[idx] if self.adjusted_feature_index is not None else idx
                node.threshold = thr

                # Recursively grow tree until maximum depth reached
                node.left = self.growTree(X_left, y_left, depth + 1)
                node.right = self.growTree(X_right, y_right, depth + 1)

        # return root node
        return node

    def predict(self, X):
        # Predict data points
        return [self._predict(inputs) for inputs in X]

    def _predict(self, inputs):
        """Predict class for a single sample."""

        # Run data through the tree and return output
        node = self.tree
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class
