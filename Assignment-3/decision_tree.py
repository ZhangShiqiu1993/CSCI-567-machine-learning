import numpy as np
from typing import List
from classifier import Classifier


class DecisionTree(Classifier):
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None
    
    def train(self, features: List[List[float]], labels: List[int]):
        # init.
        assert (len(features) > 0)
        self.feautre_dim = len(features[0])
        num_cls = np.max(labels) + 1
        
        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()
        
        return
    
    def predict(self, features: List[List[float]]) -> List[int]:
        y_pred = []
        for feature in features:
            y_pred.append(self.root_node.predict(feature))
        return y_pred
    
    def print_tree(self, node=None, name='node 0', indent=''):
        if node is None:
            node = self.root_node
        print(name + '{')
        if node.splittable:
            print(indent + '  split by dim {:d}'.format(node.dim_split))
            for idx_child, child in enumerate(node.children):
                self.print_tree(node=child, name='  ' + name + '/' + str(idx_child), indent=indent + '  ')
        else:
            print(indent + '  cls', node.cls_max)
        print(indent + '}')


class TreeNode(object):
    def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label  # majority of current node
        
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True
        
        self.dim_split = None  # the dim of feature to be splitted
        
        self.feature_uniq_split = None  # the feature to be splitted
    
    def split(self):
        def conditional_entropy(branches: List[List[int]]) -> float:
            '''
            branches: C x B array,
                      C is the number of classes,
                      B is the number of branches
                      it stores the number of
            https://piazza.com/class/jbzpcp2r9tg7b3?cid=595
            branches = [[0,4,2], [2,0,4]]
            should return 0.45
            '''
            def uncertainty(branch_group):
                p = branch_group / np.sum(branch_group)
                p[p == 0] = 1
                return -np.dot(p, np.log2(p))
            
            weight = np.sum(branches, axis=0)
            entropy = np.apply_along_axis(uncertainty, axis=0, arr=branches)
            return np.dot(weight, entropy) / np.sum(weight)
            
        
        ########################################################
        #  compute the conditional entropy
        ########################################################
        labels = np.array(self.labels)
        classes = np.unique(labels)
        C = len(classes)
        best_split = None
        max_conditional_entropy = -1
        features = np.array(self.features)
        if len(self.features[0]) == 0:
            self.splittable = False
            return
        
        for idx_dim in range(len(self.features[0])):
            ############################################################
            #  compare each split using conditional entropy
            #       find the best split
            ############################################################
            feature = features[:, idx_dim]
            branching_factor = np.unique(feature)
            B = len(branching_factor)
            branches = np.zeros((C, B))
            
            for label_index in range(C):
                label = classes[label_index]
                nodes = feature[labels == label]
                for branch_index in range(B):
                    branch = branching_factor[branch_index]
                    branches[label_index, branch_index] = np.sum(nodes == branch)
            current_conditional_entropy = conditional_entropy(branches)
            if current_conditional_entropy > max_conditional_entropy:
                max_conditional_entropy = current_conditional_entropy
                best_split = (idx_dim, feature, branching_factor)
        ############################################################
        # split the node, add child nodes
        ############################################################
        idx_dim, feature, branching_factor = best_split
        self.dim_split = idx_dim
        self.feature_uniq_split = []
        features = np.delete(features, idx_dim, axis=1)
        for branch in branching_factor:
            index = feature == branch
            new_labels = labels[index]
            new_features = features[index]
            num_cls = len(np.unique(new_labels))
            self.children.append(TreeNode(new_features.tolist(), new_labels.tolist(), num_cls))
            self.feature_uniq_split.append(branch)
        
        # split the child nodes
        for child in self.children:
            if child.splittable:
                child.split()
        
        return
    
    def predict(self, feature: List[int]) -> int:
        if self.splittable:
            # print(feature)
            idx_child = self.feature_uniq_split.index(feature[self.dim_split])
            return self.children[idx_child].predict(feature)
        else:
            return self.cls_max
