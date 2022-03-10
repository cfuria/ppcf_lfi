#Horizon
#Importing packages and initial states of the variables
from uedge import *
from rd_d3dHsm_in import *

#com.nxleg = com.nxleg*4
#com.nxcore = com.nxcore*4
#com.nycore = com.nycore*4
#com.nysol = com.nysol*4
#bbb.allocate()

import torch
from torch.distributions import Independent, MultivariateNormal, Uniform
from sbi import utils as utils
from sbi import analysis as analysis
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import pickle

#Settings
num_anom = 3
num_vals = 5
p_num = len(bbb.kye_use)
r_num = len(bbb.kye_use[0])

#Prior
prior = utils.BoxUniform(
    low = torch.ones(r_num * num_anom) * 0.1,
    high = torch.ones(r_num * num_anom) * 1.5
)

#Theta maker
def theta_maker(prior_params):
  theta_o = torch.ones(1, r_num*num_anom)
  intersections = np.ones(3)
  for i in range(3):
    def func(x):
      return -1 * prior_params[5*i+3] * math.tanh(3*(torch.tensor(x)-(prior_params[5*i]-0.3))) + prior_params[5*i+3] - 1 * prior_params[5*i+4] * math.tanh(3*(torch.tensor(x)-(prior_params[5*i]+prior_params[5*i+1]+1.2))) - prior_params[5*i+4]
    intersections[i] = fsolve(func, torch.tensor([0.6]))
  for i in range(len(com.psinormc)):
    x_param = 4 / 0.09 * (com.psinormc[i] - 0.97) - 1
    for j in range(3):
      if (x_param < intersections[j]):
        theta_o[0,r_num*j+i] = -1 * prior_params[5*j+3] * math.tanh(3*(x_param-(prior_params[5*j]-0.3))) + prior_params[5*j+2] + prior_params[5*j+3]
      else:
        theta_o[0,r_num*j+i] = prior_params[5*j+4] * math.tanh(3*(x_param-(prior_params[5*j]+prior_params[5*j+1]+1.2))) + prior_params[5*j+2] + prior_params[5*j+4]    
  return theta_o

#Simulator
from cf_rundt import *
import time
from uedge.hdf5 import *

def simulator(anom_coeffs):  
  hdf5_restore('d3d_travis.hdf5')
  bbb.restart = 1
  bbb.isbohmcalc = 3
  bbb.iflcore = 1
  bbb.pcoree = 1304606.851066112
  bbb.pcorei = 1305388.0621705272
  bbb.istewc = 0
  bbb.istiwc = 0
  bbb.istepfc = 0
  bbb.istipfc = 0
  bbb.travisv = 1
  print(anom_coeffs)
  for i in range(r_num):
    bbb.difniv[i, 0] = anom_coeffs[i]
  bbb.kyiv = anom_coeffs[r_num:2*r_num]
  bbb.kyev = anom_coeffs[2*r_num:3*r_num]

  #Second run
  bbb.ftol=1e-8 
  bbb.isbcwdt=1
  bbb.dtreal = 1e-14
  bbb.itermx=100
  bbb.exmain()

  #Rundt and steady state
  bbb.t_stop=1e2
  rundt(dtreal=1e-8,savedir="../solutions_serial")
  bbb.dtreal=1e20
  bbb.isbcwdt=0
  bbb.itermx = 100
  bbb.exmain()

  ne_u = bbb.ne[bbb.ixmp,].copy() / 10e18 # all, or just SOL (6/8 are SOL)
  te_u = bbb.te[bbb.ixmp,].copy() / bbb.ev
  ti_u = bbb.ti[bbb.ixmp,].copy() / bbb.ev
  ne_t = bbb.ne[com.nx,].copy() / 10e18
  te_t = bbb.te[com.nx,].copy() / bbb.ev
  #0:r_num-1
  measurements = np.array([ne_u, te_u, ti_u, ne_t, te_t])

  if bbb.iterm != 1:
    measurements[0] = float('nan')
  if bbb.iterm == 7:
    print("error")
  
  print(bbb.difniv)
  print(bbb.kyiv)
  print(bbb.kyev)
  return measurements

#Inference
batch_size = 1000
adj_factor = 1 
rounds = 15
learn_rate = 0.0005

import numpy as np
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
import seaborn as sns
import pandas as pd

simulator, prior = prepare_for_sbi(simulator, prior)
neural_posterior = utils.posterior_nn(model='nsf', hidden_features=120, num_transforms=10) #(80,10) original
inference = SNPE(prior=prior, density_estimator=neural_posterior)

posteriors = []
proposal = prior

#Dummy
theta_o, x_o = simulate_for_sbi(simulator, proposal=prior, num_simulations=1)
while math.isnan(x_o[0,0]):
  theta_o, x_o = simulate_for_sbi(simulator, proposal=prior, num_simulations=1) 

#Actual
theta_o_param = torch.tensor([0.24, 0.27, 0.32, 0.37, 0.15, 0.13, 0.41, 0.22, 0.13, 0.37, 0.38, 0.14, 0.46, 0.45, 0.43])
theta_o = theta_maker(theta_o_param)
x_o = simulator(theta_o)

powere = sum(bbb.feey[int(com.ixpt1)+1:int(com.ixpt2),0])
poweri = sum(bbb.feiy[int(com.ixpt1)+1:int(com.ixpt2),0])

#print("Electron power: " + str(powere))
#print("Ion power: " + str(poweri))

np.savez("n_1000p/t_17/theta_o.npz", theta_o.numpy())
np.savez("n_1000p/t_17/x_o.npz", x_o.numpy()) 
np.savez("n_1000p/t_17/psinorm.npz", com.psinormc)

for i in range(rounds):
  theta, x = simulate_for_sbi(simulator, proposal, num_simulations=batch_size*adj_factor)
  theta = theta[x[:, 0] > -100]
  x = x[x[:, 0] > -100]
  x = x[theta[:, 0] > -100]
  theta = theta[theta[:, 0] > -100]
  
  density_estimator = inference.append_simulations(theta, x, proposal=proposal).train(show_train_summary = True, use_combined_loss = True, num_atoms = 20, validation_fraction = 0.20)#, training_batch_size = 25, learning_rate = learn_rate)#, retrain_from_scratch_each_round=True)  
  posterior = inference.build_posterior(density_estimator)
  posteriors.append(posterior)
  proposal = posterior.set_default_x(x_o[0])
  learn_rate = learn_rate * 0.9
  adj_factor = 1
  
  with open("n_1000p/t_17/rd_" + str(i+1) + ".pkl", "wb") as handle:
    pickle.dump(posterior, handle)
