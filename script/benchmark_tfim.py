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
# Inverse temperature
beta = 1.0

# Specify the Hamiltonian of the QBM
# Here we consider the case where there is no model mismatch, i.e.,
# the same Hamiltonian is used to generate the date and to model the QBM.
ham_label = 0


########
# DATA #
########

match ham_label:
    
    case 0: # Transverse-field Ising model

        target_jz = rng.normal()
        target_bx = rng.normal()
        target_params = [target_jz, target_bx]
        target_ham_terms = hamiltonians.Hamiltonian_terms(n, ham_label)

        target_eta, target_expects = data.tfim_gibbs_expect(
            n, 
            target_jz, 
            target_bx, 
            beta, 
            target_ham_terms
        )


################
# QBM Taininig #
################

ham_terms = target_ham_terms
jz = rng.normal()
bx = rng.normal()
qbm_params = [jz, bx]

qbm_params, grads_hist, qre_hist = training.training_qbm(
    ham_terms,
    target_expects,
    target_eta,
    params=qbm_params
)


print(f"target parameters: {target_params}")
print(f"trained parameters: {qbm_params}")
print(f"Relative entropy: {qre_hist}")

# import matplotlib.pyplot as plt

# plt.plot(qre_hist)
# plt.show()


