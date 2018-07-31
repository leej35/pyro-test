# let's run examples in http://pyro.ai/examples/intro_part_i.html

import torch

import pyro
import pyro.distributions as dist


# Primitive Stochastic Functions

loc = 0.   # mean zero
scale = 1. # unit variance
normal = dist.Normal(loc, scale) # create a normal distribution object
x = normal.sample() # draw a sample from N(0,1)
print("sample", x)
print("log prob", normal.log_prob(x)) # score the sample from N(0,1)

x = pyro.sample("my_sample", dist.Normal(loc, scale))
print(x)



# A Simple Model

# note: 
# first, sample 'weather' from bernoulli dist 
# and then, sample temperature from one of normal distributions that depends on 
# prior info on mean and scale (params of normal dist) for each sunny and cloudy 

# weather() specifies a joint probability dist over two random variables (cloudy 
# and sunny)

def weather():
    cloudy = pyro.sample('cloudy', dist.Bernoulli(0.3))
    cloudy = 'cloudy' if cloudy.item() == 1.0 else 'sunny'
    mean_temp = {'cloudy': 55.0, 'sunny': 75.0}[cloudy]
    scale_temp = {'cloudy': 10.0, 'sunny': 15.0}[cloudy]
    temp = pyro.sample('temp', dist.Normal(mean_temp, scale_temp))
    return cloudy, temp.item()

for _ in range(3):
    print(weather())
