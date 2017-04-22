# Import stuff!
import numpy as np
import tables
import easygui
import sys
import os
import pymc3 as pm
import theano.tensor as tt

def laser_off_trials(data, num_emissions):
	
	# Make the pymc3 model
	with pm.Model() as model:
		# Dirichlet prior on the emission/spiking probabilities - 4 states
		p = pm.Dirichlet('p', np.ones(num_emissions), shape = (4, num_emissions))

		# Discrete Uniform switch times
		# Switch from detection to identity firing
		t1 = pm.DiscreteUniform('t1', lower = 10, upper = 40)
		# Duration of identity related firing
		identity_duration = pm.DiscreteUniform('identity_duration', lower = 20, upper = 100)
		# Switch from identity to palatability firing
		t2 = pm.Deterministic('t2', t1 + identity_duration)
		# Duration of palatability related firing
		palatability_duration = pm.DiscreteUniform('palatability_duration', lower = 30, upper = 120)
		# Switch from palatability firing to end
		t3 = pm.Deterministic('t3', t2 + palatability_duration)

		# Add potentials to keep the switch times from coming too close to each other
		#t_pot1 = pm.Potential('t_pot1', tt.switch(t2 - t1 >= 20, 0, -np.inf))
		#t_pot2 = pm.Potential('t_pot2', tt.switch(t3 - t2 >= 30, 0, -np.inf))
		#t_pot3 = pm.Potential('t_pot3', tt.switch(t3 - t1 >= 50, 0, -np.inf))

		# Get the actual state numbers based on the switch times
		states1 = tt.switch(tt.and_(t1 <= np.arange(250), t2 >= np.arange(250)), 1, 0)
		states2 = tt.switch(tt.and_(t2 <= np.arange(250), t3 >= np.arange(250)), 2, 0)
		states3 = tt.switch(t3 >= np.arange(250), 0, 3)
		states = states1 + states2 + states3

		# Categorical observations
		obs = pm.Categorical('obs', p = p[states], observed = data[:])

	# Inference button :D
	with model:
		tr = pm.sample(300000, init = None, step = pm.Metropolis(), njobs = 2, start = {'t1': 20, 't2': 70, 't3': 120})

	# Return the inference!
	return model, tr[250000:]

def laser_early_trials(data, num_emissions):

	# Make the pymc3 model
	with pm.Model() as model:
		# Dirichlet prior on the emission/spiking probabilities - 4 states
		p = pm.Dirichlet('p', np.ones(num_emissions), shape = (4, num_emissions))

		# Discrete Uniform switch times
		# Switch from detection to identity firing
		t1 = pm.DiscreteUniform('t1', lower = 10, upper = 40)
		# Switch from identity to palatability firing
		t2 = pm.DiscreteUniform('t2', lower = 30, upper = 130)
		# Switch from palatability firing to end
		t3 = pm.DiscreteUniform('t3', lower = 100, upper = 180)

		# Add potentials to keep the switch times from coming too close to each other
		t_pot1 = pm.Potential('t_pot1', tt.switch(t2 - t1 >= 20, 0, -np.inf))
		t_pot2 = pm.Potential('t_pot2', tt.switch(t3 - t2 >= 30, 0, -np.inf))
		t_pot3 = pm.Potential('t_pot3', tt.switch(t3 - t1 >= 50, 0, -np.inf))

		# Get the actual state numbers based on the switch times
		states1 = tt.switch(tt.and_(t1 <= np.arange(200), t2 >= np.arange(200)), 1, 0)
		states2 = tt.switch(tt.and_(t2 <= np.arange(200), t3 >= np.arange(200)), 2, 0)
		states3 = tt.switch(t3 >= np.arange(200), 0, 3)
		states = states1 + states2 + states3

		# Categorical observations
		obs = pm.Categorical('obs', p = p[states], observed = data[50:])

	# Inference button :D
	with model:
		tr = pm.sample(300000, init = None, step = pm.Metropolis(), njobs = 2, start = {'t1': 20, 't2': 70, 't3': 120})

	# Return the inference!
	return model, tr[250000:]	

def laser_middle_trials(data, num_emissions):

	# Make the pymc3 model
	with pm.Model() as model:
		# Dirichlet prior on the emission/spiking probabilities - 4 states
		p = pm.Dirichlet('p', np.ones(num_emissions), shape = (4, num_emissions))

		# Discrete Uniform switch times
		# Switch from detection to identity firing
		t1 = pm.DiscreteUniform('t1', lower = 10, upper = 40)
		# Switch from identity to palatability firing
		t2 = pm.DiscreteUniform('t2', lower = 30, upper = 100)
		# Switch from palatability firing to end
		t3 = pm.DiscreteUniform('t3', lower = 90, upper = 180)

		# Add potentials to keep the switch times from coming too close to each other
		t_pot1 = pm.Potential('t_pot1', tt.switch(t2 - t1 >= 20, 0, -np.inf))
		t_pot2 = pm.Potential('t_pot2', tt.switch(t3 - t2 >= 20, 0, -np.inf))
		t_pot3 = pm.Potential('t_pot3', tt.switch(t3 - t1 >= 40, 0, -np.inf))

		# Get the actual state numbers based on the switch times
		states1 = tt.switch(tt.and_(t1 <= np.arange(200), t2 >= np.arange(200)), 1, 0)
		states2 = tt.switch(tt.and_(t2 <= np.arange(200), t3 >= np.arange(200)), 2, 0)
		states3 = tt.switch(t3 >= np.arange(200), 0, 3)
		states = states1 + states2 + states3

		# Categorical observations
		obs = pm.Categorical('obs', p = p[states], observed = np.append(data[:70], data[120:]))

	# Inference button :D
	with model:
		tr = pm.sample(300000, init = None, step = pm.Metropolis(), njobs = 2, start = {'t1': 20, 't2': 60, 't3': 110})

	# Return the inference!
	return model, tr[250000:]	

def laser_late_trials(data, num_emissions):

	# Make the pymc3 model
	with pm.Model() as model:
		# Dirichlet prior on the emission/spiking probabilities - 4 states
		p = pm.Dirichlet('p', np.ones(num_emissions), shape = (4, num_emissions))

		# Discrete Uniform switch times
		# Switch from detection to identity firing
		t1 = pm.DiscreteUniform('t1', lower = 10, upper = 40)
		# Switch from identity to palatability firing
		t2 = pm.DiscreteUniform('t2', lower = 30, upper = 130)
		# Switch from palatability firing to end
		t3 = pm.DiscreteUniform('t3', lower = 100, upper = 180)

		# Add potentials to keep the switch times from coming too close to each other
		t_pot1 = pm.Potential('t_pot1', tt.switch(t2 - t1 >= 20, 0, -np.inf))
		t_pot2 = pm.Potential('t_pot2', tt.switch(t3 - t2 >= 20, 0, -np.inf))
		t_pot3 = pm.Potential('t_pot3', tt.switch(t3 - t1 >= 40, 0, -np.inf))

		# Get the actual state numbers based on the switch times
		states1 = tt.switch(tt.and_(t1 <= np.arange(200), t2 >= np.arange(200)), 1, 0)
		states2 = tt.switch(tt.and_(t2 <= np.arange(200), t3 >= np.arange(200)), 2, 0)
		states3 = tt.switch(t3 >= np.arange(200), 0, 3)
		states = states1 + states2 + states3

		# Categorical observations
		obs = pm.Categorical('obs', p = p[states], observed = np.append(data[:140], data[190:]))

	# Inference button :D
	with model:
		tr = pm.sample(300000, init = None, step = pm.Metropolis(), njobs = 2, start = {'t1': 20, 't2': 70, 't3': 120})

	# Return the inference!
	return model, tr[250000:]	
