""" Numpy implementations of unconstrained Euclidean metric HMC samplers. """

import numpy as np
import torch
from numpy import inf
import scipy.linalg as la
from hmc_base_pytorch import AbstractHmcSampler


class IsotropicHmcSampler(AbstractHmcSampler):
    """Standard unconstrained HMC sampler with identity mass matrix. """

    def kinetic_energy(self, pos, mom, mass, cache={}):
        if mass.shape[0]==1:
            return (0.5 * mom*mom)/mass
        else:
            mass_inv = torch.inverse(mass)
            return 0.5*((mom@mass_inv)@mom)
            #return 0.5*torch.mm(torch.mm(mom, mass_inv), mom)
            #return 0.5 * mom.dot(mass_inv).dot(mom)
            
    def simulate_dynamic(self, n_step, dt, pos, mom, mass, cache={}):
        if mass.shape[0]==1:
            print("pos:"+str(pos)+" "+str(pos.requires_grad))
            grad = self.energy_grad(pos, cache)
            print("grad1:"+str(grad))
            #print("grad2:"+str(grad))
            print("mom"+str(mom))
            mom = mom - 0.5 * dt * grad
            pos = pos + dt * (mom/mass)
            for s in range(1, n_step):
                mom = mom - dt * self.energy_grad(pos, cache)
                pos = pos + dt * (mom/mass)
            mom = mom - 0.5 * dt * self.energy_grad(pos, cache)
            return pos, mom, None
                
        else:
            mass_inv = torch.inverse(mass)
            mom = mom - 0.5 * dt * self.energy_grad(pos, cache)
            pos = pos + dt * (mom@mass_inv)
            for s in range(1, n_step):
                mom = mom - dt * self.energy_grad(pos, cache)
                pos = pos + dt * (mom@mass_inv)
            mom = mom - 0.5 * dt * self.energy_grad(pos, cache)
            return pos, mom, None

    def sample_independent_momentum_given_position(self, pos, cache={}):
        t = torch.Tensor(pos.shape[0])
        t.normal_()
        t.requires_grad=True
        
        #print("============"+str(self.dtype))
        #x = self.prng.normal(size=pos.shape[0]).astype(self.dtype)
        #print(x)
        return t

