import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

class RandomForestKernel:
    """
    Random forest kernel
    """
    def __init__(self, x_train, y_train, x_test,  m=200, max_depth=None):
        self.m = m
        self.max_depth = max_depth

        self._initKernel(x_train, y_train, x_test) # Constructs the kernel

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
        """

        """
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

    def _initKernel(self, x_train, y_train, x_test):
        """
        Initializes the random forest kernel
        """
        # First a random forest is created using the training data
        rf = RandomForestClassifier(n_estimators=self.m, bootstrap=True, max_depth=self.max_depth).fit(x_train, y_train)
        train_groupings = []
        test_groupings = []
        
        l_train = len(x_train)
        l_test = len(x_test)
        
        train_grouping = np.zeros(l_train)
        test_grouping = np.zeros(l_test)
        
        # Calculate grouping for each classifier tree in the random forest
        for clf in rf.estimators_:
            train_groupings.append(self.grouping(x_train, clf))
            test_groupings.append(self.grouping(x_test, clf))
            
        K_train = np.zeros((l_train, l_train))
        K_test = np.zeros((l_test, l_train))
        
        # t1 = 0
        # t2 = 0
        for train_grouping, test_grouping in zip(train_groupings, test_groupings):
            # st = time.time()
            # Sort the groupings to append efficiently to the kernel matrix
            temp = [(v,i) for i,v in enumerate(train_grouping)]
            temp.sort()
            train_sorted, train_indices = zip(*temp)
            
            temp = [(v,i) for i,v in enumerate(test_grouping)]
            temp.sort()
            test_sorted, test_indices = zip(*temp)
            # mt = time.time()
            # t1 += mt - st
    
            for i in range(l_train):
                j = i + 1
                train_index_i = train_indices[i]
                train_sorted_i = train_sorted[i]
                
                while j < l_train and train_sorted_i == train_sorted[j]:
                    K_train[train_index_i, train_indices[j]] += 1
                    j += 1
                
                j = 0
                while j < l_test and train_sorted_i != test_sorted[j]:
                    j += 1
                    
                while j < l_test and train_sorted_i == test_sorted[j]:
                    K_test[test_indices[j], train_index_i] += 1
                    j += 1
                    
        #     t2 += time.time() - mt
        # tic = time.time()
        
        for i in range(l_train):
            K_train[i, i] = self.m
            for j in range(i + 1, l_train):
                K_train[j, i] = K_train[i, j]
                
        # print(t1, t2, time.time() - tic)
        
        self.K_train = K_train/self.m
        # return K_train, K_test

    def transform(self, X):
        """
        From an input data matrix X returns the corresponding kernel matrix K.
        """
        return 0