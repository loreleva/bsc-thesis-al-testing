import dtree, json, sys
from threading import Thread
from decimal import *
import learner

class thread_active_learner(Thread):
	def __init__(self, oracle_tree, learner, paths):
		Thread.__init__(self)
		self.oracle_tree = oracle_tree
		self.learner = learner
		self.paths = paths

	def run(self):
		resp = self.learner.new_case(str(self.oracle_tree.root))
		node = self.oracle_tree.root.children[resp]
		while self.paths:
			while True:
				if node.action_value != None: #if the oracle has to declare an action the path is ended
					self.learner.declare_action(node.label, node.action_value)
					node = self.oracle_tree.root
					break
				else:
					resp = self.learner.query(node.label)
					node = node.children[resp]
			self.paths-=1
		self.learner.tree.final_tree()


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
active_learner = learner.learner()
thread_al = thread_active_learner(oracle_tree, active_learner, paths)
thread_al.start()


"""resp = active_learner.new_case(str(oracle_tree.root))
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
"""

#create random learner
paths = num_paths
random_learner = learner.learner(random=True)
#thread_rl = thread_random_learner(oracle_tree, random_learner, paths)
#thread_rl.start()

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

random_learner.tree.final_tree()

thread_al.join()
#thread_rl.join()

#count the number of nodes in the oracle's tree
total_nodes = oracle_tree.root.count_nodes_subtree()

print(f"{(Decimal(100)/total_nodes)*(total_nodes-dtree.count_different_nodes(oracle_tree, active_learner.tree))};{(Decimal(100)/total_nodes)*(total_nodes-dtree.count_different_nodes(oracle_tree, random_learner.tree))}")
