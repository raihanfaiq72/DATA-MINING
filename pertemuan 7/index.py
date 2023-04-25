import pandas as pd
import math
from anytree import Node, RenderTree

# Load data
data = pd.read_csv("tennis.csv")


# Calculate entropy of target class
def entropy(data):
    count = data["play"].value_counts()
    total = len(data)
    entropy = 0
    for i in count:
        prob = i / total
        entropy += -prob * math.log2(prob)
    return entropy


entropy_play = entropy(data)


# Calculate gain for each feature
def gain(data, feature):
    categories = data[feature].unique()
    gain = entropy_play
    for category in categories:
        subset = data[data[feature] == category]
        subset_entropy = entropy(subset)
        prob = len(subset) / len(data)
        gain -= prob * subset_entropy
    return gain


gain_outlook = gain(data, "outlook")
gain_temperature = gain(data, "temperature")
gain_humidity = gain(data, "humidity")
gain_windy = gain(data, "windy")

# Choose feature with highest gain as root node of decision tree
root = max(
    {
        "outlook": gain_outlook,
        "temperature": gain_temperature,
        "humidity": gain_humidity,
        "windy": gain_windy,
    }.items(),
    key=lambda x: x[1],
)[0]
root_node = Node(root)


# Recursively build decision tree
def build_tree(data, node):
    if entropy(data) == 0:
        return
    else:
        categories = data[root].unique()
        for category in categories:
            subset = data[data[root] == category]
            if entropy(subset) == 0:
                leaf_node = Node(subset["play"].iloc[0], parent=node, category=category)
            else:
                new_root = max(
                    {
                        "outlook": gain(subset, "outlook"),
                        "temperature": gain(subset, "temperature"),
                        "humidity": gain(subset, "humidity"),
                        "windy": gain(subset, "windy"),
                    }.items(),
                    key=lambda x: x[1],
                )[0]
                new_node = Node(new_root, parent=node, category=category)
                build_tree(subset, new_node)


build_tree(data, root_node)

# Print decision tree
for pre, fill, node in RenderTree(root_node):
    print(f"{pre}{node.name} ({node.category})")
