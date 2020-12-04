import dtree, math, random, re
from decimal import *

features = {}
actions = {}

class learner:

	def __init__(self, random=False):
		self.random = random
		self.tree = dtree.tree()
		self.cases = []
		self.case = []
		self.current = None

	def new_case(self, query):
		self.current = self.tree.add_root()
		resp = self.query(query)
		return resp


	def query(self, query):
		if self.current.label == None:
			self.current.add_label(query)
			if self.random:
				values = list(self.current.children)
				resp = random.choices(values, k=1)[0]
			else:
				values = list(self.current.children)
				resp = values[len(values)//2]
		else:
			if self.random:
				values = []
				for child in self.current.children:
					if self.current.children[child].action_value == None and self.current.children[child].u != 0:
						values.append(child)
				if not values:
					values = list(self.current.children)
				resp = random.choices(values, k=1)[0]
			else:
				max_val = []
				for child in self.current.children:
					if self.current.children[child].u == self.current.u:
						max_val.append(child)
				resp = random.choices(max_val, k=1)[0]

		self.case.append((query, resp,))
		self.current = self.current.children[resp]
		return resp

	def declare_action(self, action, action_value):
		self.current.add_label(action, action_value)
		self.case.append((action, action_value,))
		self.cases.append(self.case)
		self.case = []
		self.__recompute_u(self.current.parent)
		self.current = self.tree.root

	def __recompute_u(self, node):
		for child in node.children:
			node_child = node.children[child]
			if node_child.label == None:
				node_child.u = self.__entropy(node.children, child)
			elif node_child.action_value != None:
				node_child.u = 0
		maxim = 0
		for child in node.children:
			node_child = node.children[child]
			if node_child.u > maxim:
				maxim = node_child.u
		max_val = []
		for child in node.children:
			node_child = node.children[child]
			if node_child.u == maxim:
				max_val.append(node_child.u)
		node.u = random.choices(max_val, k=1)[0]
		if node.parent != None:
			self.__recompute_u(node.parent)


	def __entropy(self, domain, value):
		len_domain = len(domain)
		consistent = {"None" : gaussian(1, len_domain, len_domain)}
		values = list(domain)
		val_ind = values.index(value)
		i = val_ind-1
		label = None
		while i>=0:
			child = domain[values[i]]
			if label == None:
				if child.label == None:
					i-=1
					continue
				else:
					if child.action_value == None: #feature
						label = str(child) + f";{i}"
					else:
						label = str(child) + f":{child.action_value}"
					consistent[label] = (1-gaussian(i, val_ind, len_domain))
			else:
				if child.label == None:
					i-=1
					continue

				elif str(child) == label.split(";")[0]: #IF IS A FEATURE
					consistent[str(child)+f";{i}"] = (1-gaussian(i, val_ind, len_domain))

				elif str(child) + f":{child.action_value}" == label: #IF IS AN ACTION
					consistent[label] *= (1-gaussian(i, val_ind, len_domain))

				else:
					break
			i-=1
		
		i = val_ind+1
		label=None
		while i<len(domain):
			child = domain[values[i]]
			if label == None:
				if child.label == None:
					i+=1
					continue
				else:
					if child.action_value == None:
						label = str(child) + f";{i}"
					else:
						label = str(child) + f":{child.action_value}"
					if consistent.get(label) == None:
						consistent[label] = (1-gaussian(i, val_ind, len_domain))
					else:
						consistent[label] *= (1-gaussian(i, val_ind, len_domain))
			else:
				if child.label == None:
					i+=1
					continue

				elif str(child) == label.split(";")[0]: #IF IS A FEATURE
					consistent[str(child)+f";{i}"] = (1-gaussian(i, val_ind, len_domain))

				elif str(child) + f":{child.action_value}" == label: #IF IS AN ACTION
					consistent[label] *= (1-gaussian(i, val_ind, len_domain))

				else:
					break
			i+=1

		total = 0
		#if len(consistent) > 2:
		#	del consistent["None"]
		for cons_value in consistent:
			if cons_value == "None":
				total += consistent["None"]
			else:
				consistent[cons_value] = 1 - consistent[cons_value]
				total += consistent[cons_value]

		entropy = 0
		probabilities = {}
		for cons_value in consistent:
			prob = consistent[cons_value]/total
			probabilities[cons_value] = prob
			entropy += prob * Decimal(str(math.log(prob,2)))
		del probabilities["None"]
		maxim = max(list(probabilities.values()))
		for prob_value in probabilities:
			if probabilities[prob_value] == maxim:
				domain[value].most_likely = prob_value
				break
		return -entropy








def gaussian(start, dest, len_domain):
	return Decimal(str(math.exp(-( ((dest - start)**2) / (2*((len_domain*Decimal(str("0.10")))**2)) ) )))

