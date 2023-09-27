"""
Benchmark a QBM on the Hamiltonian dataset
"""
import argparse
import numpy as np

from qbm_quimb import hamiltonians, data, training

##########
# CONFIG #
##########

rng = np.random.default_rng(seed=1)

# CLI arguments:
parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=8, help="Number of qubits (8)")
parser.add_argument("--l", type=int, default=0, help="Label of QBM model (0)")
parser.add_argument("--lr", type=float, default=0.2, help="Learning rate (0.2)")
parser.add_argument(
    "--e", type=int, default=200, help="Number of traninig epochs (200)"
)
parser.add_argument("--er", type=int, default=1e-6, help="Error tolerance (1e-6)")

args = parser.parse_args()

n_qubits = args.n
model_label = args.l
learning_rate = args.lr
epochs = args.e
eps = args.er

########
# DATA #
########

# As an example, the Gibbs state of TF-Ising model (label=0)
# is taken to generate data (expectation values)
target_label = 0
target_ham_ops = hamiltonians.hamiltonian_operators(n_qubits, target_label)
target_params = [4.0, 4.0]
target_beta = 10.0

target_expects, target_eta = data.gibbs_expect(
    target_beta, target_params, target_ham_ops
)

#############
# QBM Model #
#############

model_ham_ops = hamiltonians.hamiltonian_operators(n_qubits, model_label)


################
# QBM Taininig #
################

initial_params = rng.normal(size=len(model_ham_ops))

qbm_params, max_grads_hist, qre_hist = training.training_qbm(
    model_ham_ops,
    target_expects,
    target_eta,
    initial_params=initial_params,
    learning_rate=learning_rate,
    epochs=epochs,
    eps=eps,
)


print(f"target parameters: {target_params}")
print(f"trained parameters: {qbm_params}")
# print(f"Relative entropy: {qre_hist}")

# import matplotlib.pyplot as plt

# plt.plot(qre_hist,'.')
# plt.show()
