import numpy as np
import multiprocessing as mp

def logp(data, p, states):
	return np.sum(np.log(p[states.astype('int'), np.tile(data.reshape(data.shape[0], 1, data.shape[1]), (1, states.shape[1], 1)).astype('int')]), axis = -1)

def E_step(data, identity, palatability, switchlim1, switchlim2, p):
	#switchpoints = np.array([[i, j] for i in range(switchlim1[0], switchlim1[1], 1) for j in range(i + switchlim2[0], switchlim2[1], 1)])
	#states = find_states(identity, palatability, switchpoints, data.shape[0])
	#for switchpoint1 in range(switchlim1[0], switchlim1[1], 1):
	#	for switchpoint2 in range(switchpoint1 + switchlim2[0], switchlim2[1], 1):
	#		states = find_states(identity, palatability, [switchpoint1, switchpoint2], data.shape[0])
	#		loglik_list.append([switchpoint1, switchpoint2, logp(data, p, states)])

	loglik_list = logp(data, p, states)
	max_loglik = np.argmax(loglik_list)
	return loglik_list[max_loglik], switchpoints[max_loglik, :]

def find_states(identity, palatability, switchpoints, data):
	#states1 = np.where(np.arange(length) <= switchpoints[0], np.zeros(length), identity*np.ones(length))
	#states = np.where(np.arange(length) <= switchpoints[1], states1, palatability*np.ones(length))
	#states1 = np.where(np.tile(np.arange(length).reshape(1, length), (switchpoints.shape[0], 1)) <= np.tile(switchpoints[:, 0].reshape(switchpoints.shape[0], 1), (1, length)), 0, identity)
	#states = np.where(np.tile(np.arange(length).reshape(1, length), (switchpoints.shape[0], 1)) <= np.tile(switchpoints[:, 1].reshape(switchpoints.shape[0], 1), (1, length)), states1, palatability)
	states1 = np.where(np.tile(np.arange(data.shape[1]).reshape(1, 1, data.shape[1]), (data.shape[0], switchpoints.shape[0], 1)) <= np.tile(switchpoints[:, 0].reshape(1, switchpoints.shape[0], 1), (data.shape[0], 1, data.shape[1])), np.zeros((data.shape[0], switchpoints.shape[0], data.shape[1])), np.tile(identity.reshape(data.shape[0], 1, 1), (1, switchpoints.shape[0], data.shape[1])))
	states = np.where(np.tile(np.arange(data.shape[1]).reshape(1, 1, data.shape[1]), (data.shape[0], switchpoints.shape[0], 1)) <= np.tile(switchpoints[:, 1].reshape(1, switchpoints.shape[0], 1), (data.shape[0], 1, data.shape[1])), states1, np.tile(palatability.reshape(data.shape[0], 1, 1), (1, switchpoints.shape[0], data.shape[1])))
	return states

def normalize_p(p):
	return p/np.tile(np.sum(p, axis = 1).reshape((p.shape[0], 1)), (1, p.shape[1]))

def fit(data, identity, palatability, iterations, threshold, switchlim1, switchlim2, num_states, num_emissions, restart):

	np.random.seed(restart)
	
	identity = identity.astype('int')
	palatability = palatability.astype('int')
	p = np.random.random((num_states, num_emissions))
	p = normalize_p(p)
	switches = []

	switchpoints = np.array([[i, j] for i in range(switchlim1[0], switchlim1[1], 1) for j in range(i + switchlim2[0], switchlim2[1], 1)])
	states = find_states(identity, palatability, switchpoints, data)
	logp_list = []
	converged = 0
	for i in range(iterations):
		switches = []
		this_logp = 0
		loglik_list = logp(data, p, states)
		max_loglik = np.argmax(loglik_list, axis = 1)
		logp_list.append(np.sum(np.max(loglik_list, axis = 1)))
		switches = switchpoints[max_loglik, :]

#		for trial in range(data.shape[0]):
#			logp_max, switches_max = E_step(data[trial], identity[trial], palatability[trial], switchlim1, switchlim2, p)
#			states = find_states(identity[trial], palatability[trial], switchpoints, data.shape[1])
#			loglik_list = logp(data[trial], p, states[trial])
#			max_loglik = np.argmax(loglik_list)			
#			this_logp += loglik_list[max_loglik]
#			switches.append(switchpoints[max_loglik, :])

#		switches = np.array(switches).astype('int')
#		logp_list.append(this_logp)

		max_states = states[np.arange(data.shape[0]), max_loglik, :] 
		p_numer = np.zeros((num_states, num_emissions))
		# Concatenate the logp maximizing state sequence and data together, and then find the counts of the (state, emission) pairs
		unique_pairs, unique_counts = np.unique(np.vstack((max_states.flatten(), data.flatten())), axis = 1, return_counts = True)
		unique_pairs = unique_pairs.astype('int')
		# Now add the unique counts to the right (state, emission) pair
		p_numer[unique_pairs[0, :], unique_pairs[1, :]] += unique_counts 
		# Normalizing these counts of emissions in every state will directly give the right p
		# Add a small number to the counts in case one of them is 0 - in that case, calculating logs gives "DivideByZeroError"
		p = normalize_p(p_numer + 1e-14)
		
#		p_numer = np.zeros((num_states, num_emissions))
#		p_denom = np.zeros((num_states, num_emissions))
#		for trial in range(data.shape[0]):
#			for emission in range(num_emissions):
#				p_numer[0, emission] += np.sum(data[trial][:switches[trial][0]] == emission)
#				p_denom[0, emission] += switches[trial][0]
#				p_numer[identity[trial], emission] += np.sum(data[trial][switches[trial][0]:switches[trial][1]] == emission)
#				p_denom[identity[trial], emission] += switches[trial][1] - switches[trial][0] 	
#				p_numer[palatability[trial], emission] += np.sum(data[trial][switches[trial][1]:] == emission)
#				p_denom[palatability[trial], emission] += data[trial].shape[0] - switches[trial][1]
#		p = p_numer/p_denom
#		Add a small number to the probabilities in case one of them is 0 - in that case, calculating logs gives "DivideByZeroError"
#		p = normalize_p(p + 1e-14)
	
		if i > 0 and np.abs(logp_list[-1] - logp_list[-2]) < threshold:
			converged = 1
			break
		
	return logp_list, p, switches, converged

def implement_EM(restarts, n_cpu, data, identity, palatability, iterations, threshold, switchlim1, switchlim2, num_states, num_emissions):
	pool = mp.Pool(processes = n_cpu)

	results = [pool.apply_async(fit, args = (data, identity, palatability, iterations, threshold, switchlim1, switchlim2, num_states, num_emissions, restart,)) for restart in range(restarts)]
	output = [result.get() for result in results]

	converged_seeds = np.array([i for i in range(len(output)) if output[i][3] == 1])
	if len(converged_seeds) == 0:
		print("Another round of {:d} seeds running as none converged the first time round".format(restarts))
		implement_EM(restarts, n_cpu, data, identity, palatability, iterations, threshold, switchlim1, switchlim2, num_states, num_emissions)
	else:
		logprobs = np.array([output[i][0][-1] for i in range(len(output))])
		max_logprob = np.argmax(logprobs[converged_seeds])
		return logprobs[converged_seeds[max_logprob]], output[converged_seeds[max_logprob]][1], output[converged_seeds[max_logprob]][2]

	

	



	
	






