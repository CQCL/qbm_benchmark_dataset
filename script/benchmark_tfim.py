"""
Benchmark a QBM on the Hamiltonian dataset    
"""
import argparse
import quimb as qu
import numpy as np

from qbm_quimb import hamiltonians, data, training

##########
# CONFIG #
##########

rng = np.random.default_rng(seed=1)
 
# Number of qubits
n = 6

########
# DATA #
########

# As an example, the Gibbs state of TF-Ising model (label=0) is take to generate data (expectation values)
target_label = 0
target_ham_ops = hamiltonians.hamiltonian_ops(n, target_label)
target_params = [1.0,0.5]
target_beta = 1.0

target_expects, target_eta = data.gibbs_expect(
    target_beta, 
    target_params,
    target_ham_ops
)
    

#############
# QBM Model #
#############

ham_label = 0
model_ham_ops = hamiltonians.hamiltonian_operators(n, ham_label)


################
# QBM Taininig #
################

initial_params = rng.normal(size=len(model_ham_ops))

qbm_params, max_grads_hist, qre_hist = training.training_qbm(
    model_ham_ops,
    target_expects,
    target_eta,
    initial_params=initial_params
)


print(f"target parameters: {target_params}")
print(f"trained parameters: {qbm_params}")
print(f"Relative entropy: {qre_hist}")

# import matplotlib.pyplot as plt

# plt.plot(qre_hist)
# plt.show()


