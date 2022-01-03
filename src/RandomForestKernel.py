import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import time

class RandomForestKernel:
    """
    Random forest kernel
    """
    def __init__(self, x_train, y_train,  m=200, max_depth=None):
        self.m = m
        self.max_depth = max_depth
        self.K_train = None

        self._initKernel(x_train, y_train) # Constructs the kernel


    def get_depth(self, clf):
        """
        Given a sklearn RandomForestClassifier tree returns a vector with the
        depth of each node and another boolean vector indicating whether it
        is a leaf node or not.
        """
        n_nodes = clf.tree_.node_count
        children_left = clf.tree_.children_left
        children_right = clf.tree_.children_right
        feature = clf.tree_.feature
        threshold = clf.tree_.threshold
        
        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
        while len(stack) > 0:
            # `pop` ensures each node is only visited once
            node_id, depth = stack.pop()
            node_depth[node_id] = depth

            # If the left and right child of a node is not the same we have a split
            # node
            is_split_node = children_left[node_id] != children_right[node_id]
            # If a split node, append left and right children and depth to `stack`
            # so we can loop through them
            if is_split_node:
                stack.append((children_left[node_id], depth + 1))
                stack.append((children_right[node_id], depth + 1))
            else:
                is_leaves[node_id] = True
        return node_depth, is_leaves


    def grouping(self, x, clf):
        """Returns a vector of indices of the grouping of each observation in a
        data matrix x using a classifier tree."""
        node_depth, is_leaves = self.get_depth(clf)
        n_nodes = clf.tree_.node_count
        children_left = clf.tree_.children_left
        children_right = clf.tree_.children_right
        feature = clf.tree_.feature
        threshold = clf.tree_.threshold
        
        # Cut at a random height
        h = max(node_depth)
        d = random.randint(0,h)
        
        grouping = np.zeros(len(x))
        
        for i in range(len(x)): #Find at which node contains each sample
            x_i = x[i]
            node = 0 # Nodes where data ends up 
            depth = 0 # Depth of the nodes

            while depth <= d: 
                if children_left[node] == -1: # If children_left (or right) is -1, the node is leaf
                    break
                else:
                    f = feature[node]
                    t = threshold[node]
                    if x_i[f] <= t:
                        node = children_left[node]
                    elif x_i[f] > t: 
                        node = children_right[node]
                depth += 1
            grouping[i] = node
        
        return grouping


    def groupingsToTrainMatrix(self, groupings):
        """Returns the training kernel matrix given the groupings in the RF."""

        l = len(groupings[0])
        K = np.zeros((l,l))

        for grouping in groupings:
            temp = [(v,i) for i,v in enumerate(grouping)]
            temp.sort()
            sorted, indices = zip(*temp)

            for i in range(l):
                j = i + 1
                while j < l and sorted[i] == sorted[j]:
                    K[indices[i],indices[j]] += 1
                    j += 1

        for i in range(l):
            K[i,i] = self.m # Diagonal of the matrix
            for j in range(i+1, l):
                K[j,i] = K[i,j]
        
        return K / self.m


    def groupingsToTestMatrix(self, groupings):
        """Returns the test kernel matrix given the groupings in the RF."""

        groupings_tr = self.train_groupings # Get training groupings

        l_tr = len(groupings_tr[0]) # Training set length
        l = len(groupings[0])

        assert l_tr >= l, "The first grouping vectors must be longer or equal than the second ones."

        K = np.zeros((l,l_tr))

        for grouping_tr, grouping in zip(groupings_tr, groupings):
            # Sort groupings and keep the indices to compare efficiently
            temp = [(v,i) for i,v in enumerate(grouping_tr)]
            temp.sort()
            sorted_tr, indices_tr = zip(*temp)

            temp = [(v,i) for i,v in enumerate(grouping)]
            temp.sort()
            sorted, indices = zip(*temp)

            for i in range(l_tr):
                j = 0
                while j < l and sorted_tr[i] != sorted[j]:
                    j += 1
                while j < l and sorted_tr[i] == sorted[j]:
                    K[indices[j], indices_tr[i]] += 1
                    j += 1
        
        return K / self.m

    def RFKernelMatrix(self, rf, data):
        """Computes the random forest kernel matrix given a random forest and a data matrix.
        This is valid for both training and test kernel matrices."""
        start = time.process_time()

        groupings = []
        l = len(data)

        for clf in rf.estimators_:
            groupings.append(self.grouping(data, clf))
        
        print("Groupings: ",time.process_time() - start, "s")
        start = time.process_time()

        if self.K_train is None:
            # Training case
            self.train_groupings = groupings # Save groupings for latter computation of test kernel matrix
            K = self.groupingsToTrainMatrix(groupings)
        else:
            # Test case
            K = self.groupingsToTestMatrix(groupings)

        print("Final K:", time.process_time() - start, "s")

        return K


    def _initKernel(self, x_train, y_train):
        """ Initializes the random forest kernel """
        start = time.process_time()
        # First a random forest is created using the training data
        rf = RandomForestClassifier(n_estimators=self.m, bootstrap=True, max_depth=self.max_depth).fit(x_train, y_train)
        self._random_forest_cf = rf # Save forest for posterior calculation of kernels
        print("Random Forest:", time.process_time() - start, "s")

        self.K_train = self.RFKernelMatrix(rf, x_train)


    def transform(self, X):
        """ From an input data matrix X returns the corresponding kernel matrix K. """
        return self.RFKernelMatrix(self._random_forest_cf, X)