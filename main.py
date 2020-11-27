import dtree, json, sys
from decimal import *
import learner as learn

with open("input.json","r") as f:
	inp=json.load(f)

#create the features and the actions
features = dtree.create_features(inp["NUM_FEATURES"], inp["MIN_DOM_FEATURES"], inp["MAX_DOM_FEATURES"])
actions = dtree.create_actions(inp["NUM_ACTIONS"], inp["MIN_DOM_ACTIONS"], inp["MAX_DOM_ACTIONS"])

#set the probability for the binomial distribution of the size of sequences of the choices
dtree.prob_size_sequence = Decimal(sys.argv[2].replace(",","."))

#generate the decision tree of the oracle
oracle_tree = dtree.tree()
oracle_tree.create_random_tree()


#number of paths to perform in the oracle's tree
num_paths = int((Decimal(str(oracle_tree.root.count_actions_subtree()))/100) * Decimal(sys.argv[1]))

paths = num_paths

#create the active learner
active_learner = learn.learner()


resp = active_learner.new_case(str(oracle_tree.root))
node = oracle_tree.root.children[resp]
while paths:
	while True:
		if node.action_value != None: #if the oracle has to declare an action the path is ended
			active_learner.declare_action(node.label, node.action_value)
			node = oracle_tree.root
			break

		else:
			resp = active_learner.query(node.label)
			node = node.children[resp]
	paths-=1


#create random learner
paths = num_paths
random_learner = learn.learner(random=True)
resp = random_learner.new_case(str(oracle_tree.root))
node = oracle_tree.root.children[resp]
while paths:
	while True:
		if node.action_value != None:
			random_learner.declare_action(node.label, node.action_value)
			node = oracle_tree.root
			break

		else:
			resp = random_learner.query(node.label)
			node = node.children[resp]
	paths-=1


#make inference on the final tree obtained by the learner
active_learner.tree.final_tree()
random_learner.tree.final_tree()

#count the number of nodes in the oracle's tree
total_nodes = oracle_tree.root.count_nodes_subtree()

print(f"{(Decimal(100)/total_nodes)*(total_nodes-dtree.count_different_nodes(oracle_tree, active_learner.tree))};{(Decimal(100)/total_nodes)*(total_nodes-dtree.count_different_nodes(oracle_tree, random_learner.tree))}")




