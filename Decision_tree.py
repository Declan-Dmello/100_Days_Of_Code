import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None,threshold=None,right=None,left=None,*,value=None):
        self.feature = feature
        self.threshold=threshold
        self.right=right
        self.left=left
        self.value = value

    def isleafnode(self):
        return self.value is not None

class DecisionTree:
    def __init__(self,*, min_samples_split =2, max_depth = 100,n_features= None):
        self.min_samples_split= min_samples_split
        self.max_depth = max_depth
        self.n_feature =n_features
        self.root = None


    def fit(self,X,y):
        self.n_feature = X.shape[1] if not self.n_feature else min(X.shape[1], self.n_feature)
        self.root = self._grow_tree(X,y)

    def _grow_tree(self,X,y, depth=0):
        n_samples , n_feats = X.shape
        n_labels = len(np.unique(y))
        #check the stopping criteria
        if depth >=self.max_depth or n_labels ==1 or n_samples<self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats , self.n_feature, replace = False)
        #find the best split
        best_feature,best_thresh = self._best_split(X,y, feat_idxs)
        left_idxs, right_idxs =  self._split(X[:,best_feature],best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feature, best_thresh, left, right)




        #create child nodes
    def _best_split(self, X, y,feat_idxs):
        best_gain = -1
        split_idx , split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:,feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                gain= self._information_gain(y, X_column, thr)

                if gain> best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr
                #calculate the info gain
        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold):
        #parent entropy
        parent_entropy = self._entropy(y)
        #create children
        left_idxs , right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) ==0:
            return 0
        n = len(y)

        #calculate the weight avg
        n_l , n_r = len(left_idxs), len(right_idxs)
        e_l , e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        #calculate the IG
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_thresh):
        left_idxs =  np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs,right_idxs


    def _entropy(self, y):
        hist  = np.bincount(y)
        ps = hist/len(y)
        return -np.sum([p * np.log(p) for p in ps if p>0])


    def _most_common_label(self,y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value



    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.isleafnode():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x,node.right)

