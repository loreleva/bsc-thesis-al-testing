import copy,dtree,math,random
from decimal import *

class learner:

	def __init__(self):
		self.features = copy.deepcopy(dtree.features)
		self.actions = copy.deepcopy(dtree.actions)
		self.tree = dtree.tree()
		self.case = {}
		self.cases = []
		self.current = None
		self.random = False

	# Active Learning
	def new_case(self, feature):
		if not self.tree.root:
			self.current = self.tree.add_root(feature)
			children = list(self.current.children)
			if not random:
				resp = children[len(children)//2]
			else:
				resp = random.choice(children)
		else:
			self.current = self.tree.root
			if not self.random:
				entropies = []
				for ch in self.current.children:
					entropies.append(self.current.children[ch].utility)
				maxim = max(entropies)
				for ch in self.current.children:
					if self.current.children[ch].utility == maxim:
						resp = ch
						break
			else:
				resp = random.choices(list(self.current.children), k=1)[0]
				while self.current.children[resp].utility ==0:
					resp = random.choices(list(self.current.children), k=1)[0]
		self.case[self.current] = resp
		self.current = self.current.children[resp]
		return resp

	def query(self, feature):
		if not self.current.variable:
			self.current.add_feature(feature)
			children = list(self.current.children)
			if not random:
				resp = children[len(children)//2]
			else:
				resp = random.choice(children)
		else:
			if not self.random:
				entropies = []
				for ch in self.current.children:
					entropies.append(self.current.children[ch].utility)
				maxim = max(entropies)
				for ch in self.current.children:
					if self.current.children[ch].utility == maxim:
						resp = ch
						break
			else:
				resp = random.choices(list(self.current.children),k=1)[0]
				while self.current.children[resp].utility ==0:
					resp = random.choices(list(self.current.children),k=1)[0]
		self.case[self.current] = resp
		self.current = self.current.children[resp]
		return resp


	def action(self, action, value):
		self.current.add_action(action, value)
		self.case[self.current] = self.current.action_value

	def end_path(self):
		self.cases.append(self.case)
		self.case={}
		self.recompute_entropies()
		self.current=None

	def recompute_entropies(self): # recompute each entropy of each node, if a node has as choice a feature, its entropy is the max of the children
		node = self.current.parent
		while node != None: # while until the root
			entropies = []
			none_count = 0
			for child in node.children:
				if node.children[child].variable == None:
					none_count+=1
			#print("\nNode: " + node.variable)
			for ch in node.children: #for each value of the node take the max entropy
				child = node.children[ch]
				if child.variable != None and child.action_value != None: # se è un'azione l'entropia è 0
					child.utility = 0
				elif child.variable == None:
					child.utility = self.compute_entropy(ch, node.children)[0]

				entropies.append(child.utility)
				#print("Value: " + str(child.value) + " Entropy: " + str(child.utility))
			node.utility=max(entropies)
			node = node.parent
	

	def final_tree(self):
		node = self.tree.root
		if node != None:
			for ch in node.children:
				child = node.children[ch]
				if child.variable == None:
					prob = self.compute_entropy(ch, node.children)[1]
					del prob["None"]
					maxim = max(list(prob.values()))
					for pr in prob:
						if prob[pr] == maxim:
							pr = pr.split(":")
							if len(pr) == 2:
								child.add_action(pr[0],int(pr[1]))
							else:
								pr = pr[0].split(";")
								child.add_feature(pr[0])
				elif child.variable != None and child.action_value == None:
					self.rec_final_tree(child)

	def rec_final_tree(self,node):
		for ch in node.children:
			child = node.children[ch]
			if child.variable == None:
				prob = self.compute_entropy(ch, node.children)[1]
				del prob["None"]
				maxim = max(list(prob.values()))
				for pr in prob:
					if prob[pr] == maxim:
						pr = pr.split(":")
						if len(pr) == 2:
							child.add_action(pr[0],int(pr[1]))
						else:
							pr = pr[0].split(";")
							child.add_feature(pr[0])
			elif child.variable != None and child.action_value == None:
				self.rec_final_tree(child)



	def compute_entropy(self, value, nodes): #calcola l'entropia di value
		global sigma
		n_feat = 0
		values = list(nodes)
		len_domain = len(values)
		i_val = values.index(value)
		i = i_val
		prob_choices = {"None" : alpha(0,(len_domain//2),len_domain)} #aggiungo sigma
		i-=1
		current = None
		while i>=0: #vado verso sinistra
			choice = nodes[values[i]] # scelta effettuata per ogni valore (none, azione, feature, ...)
			if current == None: # non ho mai incontrato choices
				if choice.variable == None: #se incontro solo none all'inizio
					pass
				else: # se incontro la prima choice
					if choice.action_value != None: # se è un'azione
						current = choice.variable + ":" +str(choice.action_value)
					else:
						current = choice.variable + ";" + str(n_feat)
						n_feat += 1
					prob_choices[current] = (1-alpha(i,i_val,len_domain))
			else:
				if choice.variable == None: #incontro un None e lo ignoro
					i-=1
					continue
				elif (choice.variable + ":" + str(choice.action_value)) == current: #incontro una choice uguale e aggiungo la sua probabilità
					prob_choices[current] *= (1-alpha(i,i_val,len_domain))
				elif (choice.variable + ":" + str(choice.action_value)) != current and choice.variable != current: #incontro un'altra choice e mi fermo
					break
			i-=1
		i = i_val+1
		current = None
		while i<len_domain: #vado verso destra
			choice = nodes[values[i]]
			if current == None: # non ho mai incontrato choices
				if choice.variable == None: #se incontro solo none all'inizio
					pass
				else: # se incontro la prima choice
					if choice.action_value != None:
						current = choice.variable +":"+ str(choice.action_value)
					else:
						current = choice.variable +";"+ str(n_feat)
						n_feat += 1
					if prob_choices.get(current) == None:
						prob_choices[current] = (1-alpha(i,i_val,len_domain))
					else:
						prob_choices[current] *= (1-alpha(i,i_val,len_domain))
			else:
				if choice.variable == None: #incontro un None e lo ignoro
					i+=1
					continue
				elif (choice.variable + ":" +str(choice.action_value)) == current: #incontro una choice uguale e aggiungo la sua probabilità
					prob_choices[current] *= (1-alpha(i,i_val,len_domain))
				elif (choice.variable + ":" + str(choice.action_value)) != current: #incontro un'altra choice e mi fermo
					break
			i+=1
		entropy = 0
		for el in prob_choices:
			if el!="None":
				prob_choices[el] = 1-prob_choices[el]
		total = Decimal(math.fsum(list(prob_choices.values())))
		for choice in prob_choices:
			prob = (prob_choices[choice] / total)
			prob = prob * Decimal(math.log(prob,2))
			entropy+=prob
		return (-entropy,prob_choices)


def alpha(start, objective, lenght_domain): 
	return Decimal(math.exp(-(Decimal(math.pow((Decimal(str(objective))-Decimal(str(start))),2)) / (2 * Decimal(math.pow(Decimal(str(lenght_domain**(1./3.))),2)) ) ) ) ) 