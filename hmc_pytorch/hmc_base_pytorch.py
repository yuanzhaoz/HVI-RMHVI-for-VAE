""" Hamiltonian dynamics based MCMC samplers. """

import logging
import numpy as np
import torch


logger = logging.getLogger(__name__)


class DynamicsError(Exception):
    """Base class for exceptions due to error in simulation of dynamics. """
    pass


class AbstractHmcSampler(object):
    """ Abstract Hamiltonian Monte Carlo sampler base class. """

    def __init__(self, energy_func, energy_grad=None, prng=None,
                 mom_resample_coeff=1., dtype=np.float64):
      
        self.energy_func = energy_func
        def gradient(pos, cache):
            #print("------calling energy function------")
            pos.retain_grad()
            x = energy_func(pos, cache)
            x.backward(retain_graph=True)
            
            g = pos.grad.clone()
            #pos.grad.zero_()
            #print("g:"+str(g))
            pos.grad.data.zero_()
            return g

        if energy_grad is None:
            self.energy_grad = gradient
        else:
            self.energy_grad = energy_grad
        
        self.prng = prng if prng is not None else np.random.RandomState()
        if mom_resample_coeff < 0 or mom_resample_coeff > 1:
                raise ValueError('Momentum resampling coefficient must be in '
                                 '[0, 1]')
        self.mom_resample_coeff = mom_resample_coeff
        self.dtype = dtype

    def kinetic_energy(self, pos, mom, mass, cache={}):
        raise NotImplementedError()

    def simulate_dynamic(self, n_step, dt, pos, mom, mass, cache={}):
        raise NotImplementedError()

    def sample_independent_momentum_given_position(self, pos, cache={}):
        raise NotImplementedError()

    def resample_momentum(self, pos, mom, cache={}):
        if self.mom_resample_coeff == 1:
            return self.sample_independent_momentum_given_position(pos, cache)
        elif self.mom_resample_coeff == 0:
            return mom
        else:
            mom_i = self.sample_independent_momentum_given_position(pos, cache)
            return (self.mom_resample_coeff * mom_i +
                    (1. - self.mom_resample_coeff**2)**0.5 * mom)

    def hamiltonian(self, pos, mom, mass, cache={}):

        return (self.energy_func(pos, cache) +
                self.kinetic_energy(pos, mom, mass, cache))

    def get_samples(self, pos, dt, n_step_per_sample, n_sample, mass, mom=None):
        #pos = pos.detach().numpy()
        n_dim = pos.shape[0]
        pos_samples, mom_samples = torch.empty((2, n_sample, n_dim))
        cache = {}
        if mom is None:
            mom = self.sample_independent_momentum_given_position(pos, cache)
        pos_samples[0], mom_samples[0] = pos, mom

        # check if number of steps specified by tuple and if so extract
        # interval bounds and check valid
        if isinstance(n_step_per_sample, tuple):
            randomise_steps = True
            step_interval_lower, step_interval_upper = n_step_per_sample
            assert step_interval_lower < step_interval_upper
            assert step_interval_lower > 0
        else:
            randomise_steps = False

        #print("========== 1st hamiltonian ==============")
        hamiltonian_c = self.hamiltonian(pos, mom, mass, cache)
        n_reject = 0

        for s in range(1, n_sample):
            #print("Sample: "+str(s))
            if randomise_steps:
                n_step_per_sample = self.prng.random_integers(
                    step_interval_lower, step_interval_upper)
            # simulate Hamiltonian dynamic to get new state pair proposal
            try:
                #print("========== simulate ==============")
                pos_p, mom_p, cache_p = self.simulate_dynamic(
                    n_step_per_sample, dt, pos_samples[s-1],
                    mom_samples[s-1], mass, cache)
                #print("========== 2nd hamiltonian ==============")
                hamiltonian_p = self.hamiltonian(pos_p, mom_p, mass, cache_p)
                proposal_successful = True
            except DynamicsError as e:
                logger.info('Error occured when simulating dynamic. '
                            'Rejecting.\n' + str(e))
                proposal_successful = False
            # Metropolis-Hastings accept step on proposed update
            if (proposal_successful and self.prng.uniform() <
                    torch.exp(hamiltonian_c - hamiltonian_p).item()):
                # accept move
                pos_samples[s], mom_samples[s], cache = pos_p, mom_p, cache_p
                hamiltonian_c = hamiltonian_p
            else:
                # reject move
                pos_samples[s] = pos_samples[s-1]
                # negate momentum on rejection to ensure reversibility
                mom_samples[s] = -mom_samples[s-1]
                n_reject += 1
            # momentum update transition: leaves momentum conditional invariant
            mom_samples[s] = self.resample_momentum(
                pos_samples[s], mom_samples[s], cache)
            if self.mom_resample_coeff != 0:
                #print("========== 3rd hamiltonian ==============")
                hamiltonian_c = self.hamiltonian(pos_samples[s],
                                                 mom_samples[s], mass, cache)

        return pos_samples, mom_samples, 1. - (n_reject * 1. / n_sample)