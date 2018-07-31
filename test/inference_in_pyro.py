# let's run examples in http://pyro.ai/examples/intro_part_ii.html

import numpy as np
import matplotlib.pyplot as plt

import torch

import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist

torch.manual_seed(101);

# ===================================
# Representing Marginal Distributions
# -----------------------------------

# assuming scale device is unreliable and gives slightly different answers
# so, let's compensate for this variability by integrating the noisy measurement
# information with a guess based on some prior knowledge about the object, 
# like its density or material properties

def scale(guess):
    # The prior over weight encodes our uncertainty about our guess
    weight = pyro.sample("weight", dist.Normal(guess, 1.0))
    # This encodes our belief about the noisiness of the scale:
    # the measurement fluctuates around the true weight
    return pyro.sample("measurement", dist.Normal(weight, 0.75))


# output of posterior will be consumed by pyro.infer.EmpiricalMarginal, which creates 
# a primitive stochastic function

posterior = pyro.infer.Importance(scale, num_samples=100)

guess = 8.5

marginal = pyro.infer.EmpiricalMarginal(posterior.run(guess))
print(marginal())


# posterior: generates a sequence of weighted execution traces given guess
# pyro.infer.EmpiricalMarginal : builds a histogram over return values from the traces, 
#                                and finally returns a sample drawn from the histogram



