import probabilities as pr
import random, copy, re

features = {}
actions = {}
prob_size_sequence = 0

class tree:

	def __init__(self, root=None):
		self.root=root

	def add_root(self,label=None):
		self.root=node(label)
		return self.root

	def __str__(self):
		res = str(self.root)
		length = len(res)+1
		if self.root:
			if self.root.children:
				res+=':\n'
				for value in self.root.children:
					res += ' '*(len(str(self.root))+1) + str(value) + "--->" + self.__rec_print(self.root.children[value], length+len(str(value))+4)
		return res
	
	def __rec_print(self, node, length):
		res = str(node)
		if node.children:
			res+=':\n'
			for value in node.children:
				res += ' '*(length+len(str(node))+1) + str(value) + "--->" + self.__rec_print(node.children[value], length+len(str(node))+4)
		elif node.action_value != None:
			res+=': ' + str(node.action_value) + '\n'
		else:
			res+='\n'
		return res

	def create_random_tree(self):
		global features, actions, prob_size_sequence
		num_features = len(features)
		prior_prob_f = 0.90
		copy_features = copy.deepcopy(features)
		copy_actions = copy.deepcopy(actions)
		root = self.add_root(random.choices(list(features), k=1)[0])
		values = copy.deepcopy(features[str(root)])
		original_size = len(values)
		del copy_features[str(root)]
		while values:
			value = values[0]

			#compute prob. size of the sequence
			res_prob_seq = pr.binomial(len(values)-1, prob_size_sequence)
			size_seq = random.choices([n+1 for n in range(len(values))], weights=[float(x) for x in res_prob_seq], k=1)[0] 
			if len(values)-size_seq < 2:
				size_seq=len(values)

			#compute prob. of a feature or of an action
			prob_f_a = pr.p_feat_action(num_features, prior_prob_f, prior_prob_f)
			res_f_a = random.choices(["f","a"], weights=[float(x) for x in prob_f_a], k=1)[0]
			
			if res_f_a=="a" or not copy_features: #if action is the outcome or there are no features
				action = random.choices(list(copy_actions), k=1)[0]
				action_value = random.choices(copy_actions[action])[0]
				copy_actions[action].remove(action_value)
				if not copy_actions[action]:
					del copy_actions[action]
				ind_val = values.index(value)
				while size_seq:
					root.children[values[ind_val]].add_label(action, action_value)
					ind_val+=1
					size_seq-=1
			
			else: #if feature is the outcome
				feature = random.choices(list(copy_features), k=1)[0]
				ind_val = values.index(value)
				while size_seq:
					root.children[values[ind_val]].add_label(feature)
					self.__rec_create_random_tree(root.children[values[ind_val]], copy.deepcopy(copy_features), actions, prior_prob_f, prob_f_a[0], num_features)
					ind_val+=1
					size_seq-=1
			values = values[ind_val:]



	def __rec_create_random_tree(self, node, features, actions, prior_prob_f, prob_f, num_features):
		global prob_size_sequence
		copy_actions = copy.deepcopy(actions)
		values = copy.deepcopy(features[str(node)])
		original_size = len(values)
		del features[str(node)]
		while values:
			value = values[0]

			#compute prob. size of the sequence
			res_prob_seq = pr.binomial(len(values)-1, prob_size_sequence)
			size_seq = random.choices([n+1 for n in range(len(values))], weights=[float(x) for x in res_prob_seq], k=1)[0] 
			if len(values)-size_seq < 2:
				size_seq=len(values)

			#compute prob. of a feature or of an action
			prob_f_a = pr.p_feat_action(num_features, prior_prob_f, prob_f)
			res_f_a = random.choices(["f","a"], weights=[float(x) for x in prob_f_a], k=1)[0]

			if res_f_a=="a" or not features: #if action is the outcome or there are no features
				action = random.choices(list(copy_actions), k=1)[0]
				action_value = random.choices(copy_actions[action])[0]
				copy_actions[action].remove(action_value)
				if not copy_actions[action]:
					del copy_actions[action]
				ind_val = values.index(value)
				while size_seq:
					node.children[values[ind_val]].add_label(action, action_value)
					ind_val+=1
					size_seq-=1
			
			else: #if feature is the outcome
				feature = random.choices(list(features), k=1)[0]
				ind_val = values.index(value)
				while size_seq:
					node.children[values[ind_val]].add_label(feature)
					self.__rec_create_random_tree(node.children[values[ind_val]], copy.deepcopy(features), actions, prior_prob_f, prob_f, num_features)
					ind_val+=1
					size_seq-=1
			values = values[ind_val:]

	def final_tree(self):
		root = self.root
		if root.label == None:
			return
		for child in root.children:
			self.__rec_final_tree(root.children[child])


	def __rec_final_tree(self, node):
		if node.label == None:
			if len(node.most_likely.split(";")) == 2: #feature
				node.add_label(node.most_likely.split(";")[0])
			else:
				action = node.most_likely.split(":")
				action_value = action[1]
				node.add_label(action[0], int(action_value))
		elif node.action_value == None:
			for child in node.children:
				self.__rec_final_tree(node.children[child])









class node:

	def __init__(self, label=None, parent=None):
		global features, actions
		self.u=0
		self.label=label
		self.children={}
		self.action_value=None
		self.parent = parent
		self.most_likely = None

		if self.label != None and self.action_value == None: #if is a feature, for each value add children nodes
			self.__add_children()

	def __add_children(self): #create children nodes of the feature
		for value in features[self.label]:
			self.children[value] = node(None, self)

	def add_label(self, label, action_value=None): #assign the label to the node, if is a feature create the children nodes
		self.label = label
		if action_value:
			self.action_value=action_value
		else:
			self.__add_children()

	def __str__(self):
		return str(self.label)

	def count_nodes_subtree(self):
		res = 1
		for value in self.children:
			res+=self.children[value].count_nodes_subtree()
		return res

	def count_actions_subtree(self):
		res = 0
		if self.label != None and self.action_value != None:
			res+=1
		else:
			if self.children:
				for child in self.children:
					res+=self.children[child].count_actions_subtree()
		return res	 

def create_features(numb, minim, maxim):
	global features
	prob = pr.binomial(maxim-minim+1, 0.50)
	sizes = [x for x in range(minim,maxim+1)]
	for n in range(numb):
		prob = pr.binomial(maxim-minim, 0.50)
		size = random.choices(sizes, weights=[float(x) for x in prob], k=1)[0]
		features[f"feature{n+1}"] = [x+1 for x in range(size)]
	return features

def create_actions(numb, minim, maxim):
	global actions
	prob = pr.binomial(maxim-minim+1, 0.50)
	sizes = [x for x in range(minim,maxim+1)]
	for n in range(numb):
		prob = pr.binomial(maxim-minim, 0.50)
		size = random.choices(sizes, weights=[float(x) for x in prob], k=1)[0]
		actions[f"action{n+1}"] = [x+1 for x in range(size)]
	return actions

def count_different_nodes(first, second):
	res = 0
	if str(first.root) != str(second.root):
		res+=first.root.count_nodes_subtree()
	else:
		for value in first.root.children:
			res+=rec_count_different_nodes(first.root.children[value], second.root.children[value])
	return res

def rec_count_different_nodes(first, second):
	res = 0
	if first.label != second.label:
		res+=first.count_nodes_subtree()
	else:
		if first.children: #if they are features
			for value in first.children:
				res+=rec_count_different_nodes(first.children[value], second.children[value])
	return res






