import random, collections, copy, math
from decimal import *
import prob_distributions as prob_d

ndomains=0

features={}
actions={}
val_prob_seq = 0


class tree:

	def __init__(self):
		self.root = None

	def add_root(self, variable):
		root = node(variable, None, None)
		self.root = root
		return root

	def __str__(self):
		if self.root != None:
			res = self.root.variable + ":\n"
			for x in self.root.children:
				res = res + self.__rec_print(self.root.children[x],len(self.root.variable)+1)
		else:
			res="None"
		return res

	def __rec_print(self, node, length):
		res = ' '*length + str(node.value) + " ---> " + str(node.variable) 
		if not node.variable:
			return res+"\n"
		res = res + ": "
		if not node.children:
			res = res + str(node.action_value) + "\n"
		else:
			res = res+"\n"
			for x in node.children:
				res = res + self.__rec_print(node.children[x],length+len(node.variable)+len(str(node.value))+7)
		return res

	def __eq__(self, other):
		f_root=self.root
		s_root=other.root
		if f_root == s_root:
			return True
		elif (f_root == None and s_root != None) or (f_root != None and s_root == None):
			return False
		elif not f_root.variable == s_root.variable:
			return False
		else:
			for child in f_root.children:
				if not self.__rec_eq(f_root.children[child], s_root.children[child]):
					return False
		return True

	def __rec_eq(self, f_node, s_node):
		if not f_node.variable == s_node.variable:
			return False
		else:
			if f_node.action_value != None:
				return True
			else:
				for child in f_node.children:
					if not self.__rec_eq(f_node.children[child], s_node.children[child]):
						return False
		return True





class node:

	def __init__(self, variable, value=None, parent=None):
		global features
		self.variable=variable
		self.value=value
		self.action_value=None
		self.utility=None
		self.parent=parent
		if variable:
			children={}
			for val in features[variable]:
				children[val] = node(None, val, self)
			self.children=children
		else:
			self.children=None

	def add_feature(self, variable):
		global features
		self.variable=variable
		children={}
		for val in features[self.variable]:
			children[val] = node(None, val, self)
		self.children = children

	def add_action(self, action, value):
		self.variable = action
		self.action_value = value




def create_features(num, minim, maxim):
	global features
	p_numdom_features = prob_d.binomial(maxim-minim, 0.5)
	dict_prob = {}
	for prob in p_numdom_features: # create the dictionary with value,prob
		dict_prob[minim] = prob
		minim+=1
	for n in range(num): #create num features, by picking the size randomly on the prob distribution
		n+=1
		extracted = random.choices([val for val in dict_prob], weights=tuple([float(dict_prob[prob]) for prob in dict_prob]), k=1)
		features[f"feature{n}"] = [x+1 for x in range(extracted[0])]
	return features

def create_actions(num, minim, maxim):
	global actions
	p_numdom_actions = prob_d.binomial(maxim-minim, 0.5)
	dict_prob = {}
	for prob in p_numdom_actions: # create the dictionary with value,prob
		dict_prob[minim] = prob
		minim+=1
	for n in range(num): #create num actions, by picking the size randomly on the prob distribution
		n+=1
		extracted = random.choices([val for val in dict_prob], weights=tuple([float(dict_prob[prob]) for prob in dict_prob]), k=1)
		actions[f"action{n}"] = [x+1 for x in range(extracted[0])]
	return actions


def create_random_tree(features, actions, prior_prob_f):
	global val_prob_seq
	temp_actions = copy.deepcopy(actions)
	size_f_dom = len(features)
	prior_prob_f = Decimal(str(prior_prob_f))
	new_tree=tree()
	root = random.choices(list(features),k=1)[0]
	root_values = features[root]
	root = new_tree.add_root(root)
	del features[root.variable]
	orig_size = len(root_values)
	while root_values:
		# choose the value from the domain of the variable
		val = root_values[0]
		prob_f_a = prob_d.p_feat_action(size_f_dom, prior_prob_f, prior_prob_f)
		if not features:
			f_a = "a"
		else:
			#choose if use a feature or an action
			f_a = random.choices(["f","a"], weights=(float(prob_f_a[0]),float(prob_f_a[1])), k=1)[0]

		if f_a == "a":
			#choose an action,value and remove them from the temp dict
			action = random.choices(list(temp_actions), k=1)[0] #choice uniformly an action
			action_value = random.choices(temp_actions[action], k=1)[0] #choice uniformly a value
			val_ind = root_values.index(val)
			if len(root_values)< (Decimal(str(1/4))*orig_size):
				num_seq=len(root_values)-val_ind
			else:
				#pick, upon the distribution, the number of sequence, from the remaining
				prob_seq = prob_d.binomial((len(root_values)-root_values.index(val))-2, val_prob_seq) #binomial distribution for the numb of sequence
				num_seq = random.choices([x+1 for x in range(len(prob_seq))], weights=[float(n) for n in prob_seq], k=1)[0] #choice, by the distr., the num of sequence
			i = num_seq
			while i:
				root.children[root_values[val_ind]].add_action(action, action_value)
				val_ind+=1
				i-=1
			val_ind = root_values.index(val)
			del root_values[val_ind:val_ind+num_seq]
		else: #feature
			feature = random.choices(list(features),k=1)[0]
			val_ind = root_values.index(val)
			if len(root_values)< (Decimal(str(1/4))*orig_size):
				num_seq=len(root_values)-val_ind
			else:
				prob_seq = prob_d.binomial((len(root_values)-root_values.index(val))-2, val_prob_seq) #binomial distribution for the num of sequence
				num_seq = random.choices([x+1 for x in range(len(prob_seq))], weights=[float(n) for n in prob_seq], k=1)[0] #choice, by the distr., the num of sequence
			i = num_seq
			while i:
				root.children[root_values[val_ind]].add_feature(feature)
				rec_create_random_tree(root.children[root_values[val_ind]], copy.deepcopy(features), actions, prior_prob_f, prob_f_a[0], size_f_dom)
				val_ind+=1
				i-=1
			val_ind = root_values.index(val)
			del root_values[val_ind:val_ind+num_seq]
			del features[feature]
	return new_tree


def rec_create_random_tree(node, features, actions, prior_prob_f, prob_f, size_f_dom):
	temp_actions = copy.deepcopy(actions)
	node_values = features[node.variable]
	del features[node.variable]
	orig_size = len(node_values)
	while node_values:
		val = node_values[0]
		prob_f_a = prob_d.p_feat_action(size_f_dom, prior_prob_f, prob_f)
		if not features:
			f_a="a"
		else:
			f_a = random.choices(["f","a"], weights=(float(prob_f_a[0]),float(prob_f_a[1])), k=1)[0]
		if f_a == "a":
			action = random.choices(list(temp_actions), k=1)[0] #choice uniformly an action
			action_value = random.choices(temp_actions[action], k=1)[0] #choice uniformly a value
			val_ind = node_values.index(val)
			if len(node_values)< (Decimal(str(1/4))*orig_size):
				num_seq=len(node_values)-val_ind
			else:
				prob_seq = prob_d.binomial((len(node_values)-node_values.index(val))-1, val_prob_seq) #binomial distribution for the num of sequence
				num_seq = random.choices([x+1 for x in range(len(prob_seq))], weights=[float(n) for n in prob_seq], k=1)[0] #choice, by the distr., the num of sequence
			i = num_seq
			while i:
				node.children[node_values[val_ind]].add_action(action, action_value)
				val_ind+=1
				i-=1
			val_ind = node_values.index(val)
			del node_values[val_ind:val_ind+num_seq]
		else: #feature
			feature = random.choices(list(features),k=1)[0]
			val_ind = node_values.index(val)
			if len(node_values)< (Decimal(str(1/4))*orig_size):
				num_seq=len(node_values)-val_ind
			else:
				prob_seq = prob_d.binomial((len(node_values)-node_values.index(val))-1, val_prob_seq) #binomial distribution for the num of sequence
				num_seq = random.choices([x+1 for x in range(len(prob_seq))], weights=[float(n) for n in prob_seq], k=1)[0] #choice, by the distr., the num of sequence
			i = num_seq
			while i:
				node.children[node_values[val_ind]].add_feature(feature)
				rec_create_random_tree(node.children[node_values[val_ind]], copy.deepcopy(features), actions, prior_prob_f, prob_f_a[0], size_f_dom)
				val_ind+=1
				i-=1
			val_ind = node_values.index(val)
			del node_values[val_ind:val_ind+num_seq]
			del features[feature]


def count_errors(f_tree, s_tree): #f_tree oracle
	f_root=f_tree.root
	s_root=s_tree.root
	total_nodes=1
	diff_nodes=0
	if (s_root == None):
		diff_nodes +=1
		rem = rec_count_nodes(f_root)
		return (total_nodes+rem, diff_nodes+rem)
	elif not f_root.variable == s_root.variable:
		diff_nodes+=1
		rem = rec_count_nodes(f_root)
		return (total_nodes+rem, diff_nodes+rem)
	for child in f_root.children:
		res = rec_count_errors(f_root.children[child],s_root.children[child])
		total_nodes += res[0]
		diff_nodes += res[1]
	return (total_nodes, diff_nodes)

def rec_count_errors(f_node, s_node):
	total_nodes = 1
	diff_nodes = 0
	if not f_node.variable == s_node.variable:
		diff_nodes += 1
		rem = rec_count_nodes(f_node)
		return (total_nodes+ rem, diff_nodes+rem)
	else:
		if not f_node.children: #azione
			if f_node.action_value != s_node.action_value:
				diff_nodes +=1
		else:
			for child in f_node.children:
				res = rec_count_errors(f_node.children[child],s_node.children[child])
				total_nodes += res[0]
				diff_nodes += res[1]
	return (total_nodes, diff_nodes)

def rec_count_nodes(f_node):
	total_nodes = 0
	if f_node.children:
		for child in f_node.children:
			total_nodes += 1
			total_nodes += rec_count_nodes(f_node.children[child])
	return total_nodes

def count_decisions(node):
	count = 0
	if node.action_value != None:
		count +=1
	else:
		for ch in node.children:
			count += count_decisions(node.children[ch])
	return count

#--------------------------------------------------------------------------------------------------------------


def frange(x, max, step):
	while x<=max:
		yield float(x)
		x+=step

#--------------------------------------------------------------------------------------------------------------

