import math
from decimal import *

def binomial(n,p):
	p = Decimal(str(p))
	res = []
	for k in range(n+1):
		comb = Decimal(str(math.factorial(n)))/(math.factorial(k)*math.factorial(n-k))
		val = comb * (p**k) * ((1-p)**(n-k))
		res.append(val)
	return res

def p_feat_action(num_features, prior_prob_f, prob_f): #prob to choice a feature or an action
	prob_f = Decimal(str(prob_f))
	prob_f = prob_f - ((Decimal(str(prior_prob_f))) * Decimal(str(1/num_features)))
	return (prob_f,1-prob_f)