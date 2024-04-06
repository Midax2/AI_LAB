from collections import defaultdict
import numpy as np
from node_solution import Node

class DecisionTree:
    def __init__(self, params):
        self.root_node = Node()
        self.params = defaultdict(lambda: None, params)


    def train(self, X, y):
        self.root_node.train(X, y, self.params)

    def evaluate(self, X, y):
        predicted = self.predict(X)
        predicted = [round(p) for p in predicted]
        print(f"Accuracy: {round(np.mean(predicted==y),2)}")

    def predict(self, X):
        prediction = []
        for x in X:
            prediction.append(self.root_node.predict(x))
        return prediction

