import json, copy, sys, dtree, random
import prob_distributions as pr
import learning as learn
from decimal import *

with open("input.json","r") as f:
	inp=json.load(f)

#creates the features and the actions
features = dtree.create_features(inp["NUM_FEATURES"], inp["MIN_DOM_FEATURES"], inp["MAX_DOM_FEATURES"])
actions = dtree.create_actions(inp["NUM_ACTIONS"], inp["MIN_DOM_ACTIONS"], inp["MAX_DOM_ACTIONS"])

#set the probability for the binomial distribution of the size of sequences of the choices
dtree.val_prob_seq = Decimal(sys.argv[2].replace(",","."))

#generate the decision tree of the oracle
oracle_tree = dtree.create_random_tree(copy.deepcopy(features), copy.deepcopy(actions), 0.90)

#count the number of nodes in the oracle's tree
num_nodes = dtree.rec_count_nodes(oracle_tree.root)+1

#count the number of leaves of the tree
num_decisions = dtree.count_decisions(oracle_tree.root)

#create the learner
learner = learn.learner()

#number of paths to perform in the oracle's tree
num_paths = int(Decimal(str(num_decisions/100))*int(sys.argv[1]))

paths = num_paths

root = oracle_tree.root
while paths:
	#create a new case
	resp = learner.new_case(root.variable)
	node = root.children[resp]

	while 1:
		if node.action_value != None: #if the oracle has to declare an action the path is ended
			learner.action(node.variable, node.action_value)
			learner.end_path()
			break
		else:
			resp = learner.query(node.variable)
			node = node.children[resp]
	paths-=1

random_learner = learn.learner()
random_learner.random = True

paths = num_paths

#random learner
while paths:
	#create a new case
	resp = random_learner.new_case(root.variable)
	node = root.children[resp]

	while 1:
		if node.action_value != None: #if the oracle has to declare an action the path is ended
			random_learner.action(node.variable, node.action_value)
			random_learner.end_path()
			break
		else:
			resp = random_learner.query(node.variable)
			node = node.children[resp]
	paths-=1

paths = num_paths

#print(f"Oracle's tree:\n{oracle_tree}\n\nLearner's tree:\n{learner.tree}\n\nRandom learner's tree:\n{random_learner.tree}")

#print(f"Learner's tree:\n{learner.tree}\nRandom learner's tree:\n{random_learner.tree}")

#make inference on the final tree obtained by the learner
learner.final_tree()
random_learner.final_tree()

#print(f"\n\nOracle's tree: \n{oracle_tree}\nFinal learner's tree: \n{learner.tree}\nRandom learner's tree: \n{random_learner.tree}")

#count the number of different nodes between the oracle's tree and the learner's tree
res = dtree.count_errors(oracle_tree, learner.tree)

random_res = dtree.count_errors(oracle_tree, random_learner.tree)

#print(f"\n\nTotal nodes={res[0]}\nTotal decisions={num_decisions}\nDifferent nodes={res[1]}\nNumber cases={num_paths}\nCorrectness rate={Decimal(str(100/num_nodes))*(num_nodes-res[1])}")

print(f"{Decimal(str(100/num_nodes))*(num_nodes-res[1])};{Decimal(str(100/num_nodes))*(num_nodes-random_res[1])}")

