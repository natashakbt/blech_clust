import numpy as np
import multiprocessing as mp

# Set random number seed
np.random.seed(0)

def logp(data, p, states):
	return np.sum(np.log(p[states.astype('int'), np.tile(data.reshape(1, data.shape[0]), (states.shape[0], 1)).astype('int')]), axis = 1)

def E_step(data, identity, palatability, switchlim1, switchlim2, p):
	switchpoints = np.array([[i, j] for i in range(switchlim1[0], switchlim1[1], 1) for j in range(i + switchlim2[0], switchlim2[1], 1)])
	states = find_states(identity, palatability, switchpoints, data.shape[0])
	#for switchpoint1 in range(switchlim1[0], switchlim1[1], 1):
	#	for switchpoint2 in range(switchpoint1 + switchlim2[0], switchlim2[1], 1):
	#		states = find_states(identity, palatability, [switchpoint1, switchpoint2], data.shape[0])
	#		loglik_list.append([switchpoint1, switchpoint2, logp(data, p, states)])

	loglik_list = logp(data, p, states)
	max_loglik = np.argmax(loglik_list)
	return loglik_list[max_loglik], switchpoints[max_loglik, :]

def find_states(identity, palatability, switchpoints, length):
	#states1 = np.where(np.arange(length) <= switchpoints[0], np.zeros(length), identity*np.ones(length))
	#states = np.where(np.arange(length) <= switchpoints[1], states1, palatability*np.ones(length))
	states1 = np.where(np.tile(np.arange(length).reshape(1, length), (switchpoints.shape[0], 1)) <= np.tile(switchpoints[:, 0].reshape(switchpoints.shape[0], 1), (1, length)), 0, identity)
	states = np.where(np.tile(np.arange(length).reshape(1, length), (switchpoints.shape[0], 1)) <= np.tile(switchpoints[:, 1].reshape(switchpoints.shape[0], 1), (1, length)), states1, palatability)
	return states

def normalize_p(p):
	return p/np.tile(np.sum(p, axis = 1).reshape((p.shape[0], 1)), (1, p.shape[1]))

def fit(data, identity, palatability, iterations, threshold, switchlim1, switchlim2, num_states, num_emissions):
	
	identity = identity.astype('int')
	palatability = palatability.astype('int')
	p = np.random.random((num_states, num_emissions))
	p = normalize_p(p)
	switches = []

	logp_list = []
	for i in range(iterations):
		switches = []
		this_logp = 0
		for trial in range(data.shape[0]):
			logp_max, switches_max = E_step(data[trial], identity[trial], palatability[trial], switchlim1, switchlim2, p)
			this_logp += logp_max
			switches.append(switches_max)

		switches = np.array(switches).astype('int')
		logp_list.append(this_logp)

		p_numer = np.zeros((num_states, num_emissions))
		p_denom = np.zeros((num_states, num_emissions))
		for trial in range(data.shape[0]):
			for emission in range(num_emissions):
				p_numer[0, emission] += np.sum(data[trial][:switches[trial][0]] == emission)
				p_denom[0, emission] += switches[trial][0]
				p_numer[identity[trial], emission] += np.sum(data[trial][switches[trial][0]:switches[trial][1]] == emission)
				p_denom[identity[trial], emission] += switches[trial][1] - switches[trial][0] 	
				p_numer[palatability[trial], emission] += np.sum(data[trial][switches[trial][1]:] == emission)
				p_denom[palatability[trial], emission] += data[trial].shape[0] - switches[trial][1]
		p = p_numer/p_denom
		# Add a small number to the probabilities in case one of them is 0 - in that case, calculating logs gives "DivideByZeroError"
		p = normalize_p(p + 1e-14)
	
		if i > 0 and np.abs(logp_list[-1] - logp_list[-2]) < threshold:
			break
	return logp_list, p, switches

def implement_EM(restarts, n_cpu, data, identity, palatability, iterations, threshold, switchlim1, switchlim2, num_states, num_emissions):
	pool = mp.Pool(processes = n_cpu)

	results = [pool.apply_async(fit, args = (data, identity, palatability, iterations, threshold, switchlim1, switchlim2, num_states, num_emissions,)) for restart in range(restarts)]
	output = [result.get() for result in results]

	logprobs = np.array([output[i][0][-1] for i in range(len(output))])
	max_logprob = np.argmax(logprobs)

	return logprobs[max_logprob], output[max_logprob][1], output[max_logprob][2]

	

	



	
	






