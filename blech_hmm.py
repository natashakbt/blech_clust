# Import stuff
from yahmm import *
import numpy as np
from scipy.stats import poisson
import multiprocessing as mp

# Define the Poisson distribution by sub-classing yahmm's Distribution class
class PoissonDistribution(Distribution):
    def __init__(self, mu):
        self.name = 'PoissonDistribution'
        self.parameters = [mu]
	self.summaries = []

    def log_probability(self, sample):
        return np.log(poisson.pmf(sample, self.parameters[0]))

    def from_sample(self, items, weights = None, inertia = 0.0):
	items = np.asarray(items)
        if weights is None:
            weights = np.ones_like(items, dtype=float)
	# Get the new mean while respecting inertia
        self.parameters[0] = np.mean(items*weights)*(1.0 - inertia) + self.parameters[0]*inertia

    def sample(self):
        return poisson.rvs(self.parameters[0], size = 1)[0]

    def from_summaries(self, inertia = 0.0):
	items, weights = self.summaries
	items = np.asarray(items)
	#self.from_sample(self.summaries, inertia = inertia)
        if weights is None:
            weights = np.ones_like(items, dtype=float)
	self.parameters[0] = np.mean(items*weights)*(1.0 - inertia) + self.parameters[0]*inertia
	self.summaries = []


    def summarize(self, items, weights=None):
	# If no previously stored summaries, just store the incoming data
	if len( self.summaries ) == 0:
		self.summaries = [items, weights]

	# Otherwise, append the items and weights
	else:
		prior_items, prior_weights = self.summaries
		items = np.concatenate( [prior_items, items] )

		# If even one summary lacks weights, then weights can't be assigned
		# to any of the points.
		if weights is not None:
			weights = np.concatenate( [prior_weights, weights] )

		self.summaries = [items, weights]

   
'''def poisson_hmm_implement(min_states, max_states, max_iter, threshold, seeds, n_cpu, spikes):

	hmm_results = {}
	for i in range(min_states, max_states + 1):
'''		
	

def poisson_hmm_implement(n_states, max_iter, threshold, seeds, n_cpu, spikes, on_trials, edge_inertia, dist_inertia):

	# Create a pool of asynchronous n_cpu processes running poisson_hmm() - no. of processes equal to seeds
	pool = mp.Pool(processes = n_cpu)
	results = [pool.apply_async(poisson_hmm, args = (n_states, max_iter, threshold, spikes, seed, on_trials, edge_inertia, dist_inertia,)) for seed in range(seeds)]
	output = [p.get() for p in results]
	# Find the process that ended up with the highest log likelihood, and return it as the solution
	maximum = float('inf')*(-1)
	maximum_pos = 0
	for i in range(len(output)):
		if output[i][1] >= maximum and output[i][0] <= max_iter-2:
			maximum = output[i][1]
			maximum_pos = i
	
	if maximum == float('inf')*(-1):
		return 0
	else:	
		return output[maximum_pos]	

def multinomial_hmm_implement(n_states, max_iter, threshold, seeds, n_cpu, spikes, on_trials, edge_inertia, dist_inertia):

	# Create a pool of asynchronous n_cpu processes running multinomial_hmm() - no. of processes equal to seeds
	pool = mp.Pool(processes = n_cpu)
	results = [pool.apply_async(multinomial_hmm, args = (n_states, max_iter, threshold, spikes, seed, on_trials, edge_inertia, dist_inertia,)) for seed in range(seeds)]
	output = [p.get() for p in results]
	# Find the process that ended up with the highest log likelihood, and return it as the solution
	maximum = float('inf')*(-1)
	maximum_pos = 0
	for i in range(len(output)):
		if output[i][1] >= maximum and output[i][0] <= max_iter-2:
			maximum = output[i][1]
			maximum_pos = i
	
	if maximum == float('inf')*(-1):
		return 0
	else:	
		return output[maximum_pos]	


def poisson_hmm(n_states, max_iter, threshold, spikes, seed, on_trials, edge_inertia, dist_inertia):

	# Make a yahmm Model object
	model = Model('%i' % seed) 
	states = []
	# Make a yahmm multivariate distribution object and represent every unit with a Poisson distribution - 1 for each state
	for i in range(n_states):
		#emission_slice = (int((float(i)/n_states)*spikes.shape[1]), int((float(i+1)/n_states)*spikes.shape[1]))
		#initial_emissions = np.mean(spikes[on_trials, emission_slice[0]:emission_slice[1], :], axis = (0, 1))*(np.random.random())
		states.append(State(MultivariateDistribution([PoissonDistribution(np.random.rand()) for unit in range(spikes.shape[2])]), name = 'State%i' % (i+1)))
		
	model.add_states(states)
	# Add transitions from model.start to each state (equal probabilties)
	for state in states:
		model.add_transition(model.start, state, float(1.0/len(states)))

	# Add transitions between the states - 0.97 is the probability of not transitioning in every state
	for i in range(n_states):
		not_transitioning_prob = (0.999-0.95)*np.random.random() + 0.95
		for j in range(n_states):
			if i==j:
				model.add_transition(states[i], states[j], not_transitioning_prob)
			else:
				model.add_transition(states[i], states[j], float((1.0 - not_transitioning_prob)/(n_states - 1)))
	
	# Bake the model
	model.bake()

	# Train the model only on the trials indicated by on_trials
	model.train(spikes[on_trials, :, :], algorithm = 'baum-welch', stop_threshold = threshold, max_iterations = max_iter, edge_inertia = edge_inertia, distribution_inertia = dist_inertia)
	log_prob = [model.log_probability(spikes[i, :, :]) for i in on_trials]
	log_prob = np.sum(log_prob)

	# Set up things to return the parameters of the model - the state emission and transition matrix 
	state_emissions = []
	state_transitions = np.exp(model.dense_transition_matrix())
	for i in range(n_states):
		state_emissions.append([model.states[i].distribution.parameters[0][j].parameters[0] for j in range(spikes.shape[2])])
	state_emissions = np.array(state_emissions)

	# Get the posterior probability sequence to return
	posterior_proba = []
	for i in range(spikes.shape[0]):
		c, d = model.forward_backward(spikes[i, :, :])
		posterior_proba.append(d)
	posterior_proba = np.exp(np.array(posterior_proba))

	return model.iterations, log_prob, 2*((n_states)**2 + n_states*spikes.shape[2]) - 2*log_prob, (np.log(len(on_trials)*spikes.shape[1]))*((n_states)**2 + n_states*spikes.shape[2]) - 2*log_prob, state_emissions, state_transitions, posterior_proba
	
def multinomial_hmm(n_states, max_iter, threshold, spikes, seed, on_trials, edge_inertia, dist_inertia):

	# Make a yahmm Model object
	model = Model('%i' % seed) 
	states = []
	# Make a yahmm Discrete distribution object with emissions = range(n_units + 1) - 1 for each state
	n_units = int(np.max(spikes))
	for i in range(n_states):
		dist_dict = {}
		prob_list = np.random.random(n_units + 1)
		prob_list = prob_list/np.sum(prob_list)
		for unit in range(n_units + 1):
			dist_dict[unit] = prob_list[unit]	
		states.append(State(DiscreteDistribution(dist_dict), name = 'State%i' % (i+1)))

	model.add_states(states)
	# Add transitions from model.start to each state (equal probabilties)
	for state in states:
		model.add_transition(model.start, state, float(1.0/len(states)))

	# Add transitions between the states - 0.95-0.999 is the probability of not transitioning in every state
	for i in range(n_states):
		not_transitioning_prob = (0.999-0.95)*np.random.random() + 0.95
		for j in range(n_states):
			if i==j:
				model.add_transition(states[i], states[j], not_transitioning_prob)
			else:
				model.add_transition(states[i], states[j], float((1.0 - not_transitioning_prob)/(n_states - 1)))

	# Bake the model
	model.bake()

	# Train the model only on the trials indicated by on_trials
	model.train(spikes[on_trials, :], algorithm = 'baum-welch', stop_threshold = threshold, max_iterations = max_iter, edge_inertia = edge_inertia, distribution_inertia = dist_inertia)
	log_prob = [model.log_probability(spikes[i, :]) for i in on_trials]
	log_prob = np.sum(log_prob)

	# Set up things to return the parameters of the model - the state emission dicts and transition matrix 
	state_emissions = []
	state_transitions = np.exp(model.dense_transition_matrix())
	for i in range(n_states):
		state_emissions.append(model.states[i].distribution.parameters[0])

	# Get the posterior probability sequence to return
	posterior_proba = np.zeros((spikes.shape[0], spikes.shape[1], n_states))
	for i in range(spikes.shape[0]):
		c, d = model.forward_backward(spikes[i, :])
		posterior_proba[i, :, :] = np.exp(d)
	
	return model.iterations, log_prob, 2*((n_states)**2 + n_states*(n_units + 1)) - 2*log_prob, (np.log(len(on_trials)*spikes.shape[1]))*((n_states)**2 + n_states*(n_units + 1)) - 2*log_prob, state_emissions, state_transitions, posterior_proba	
