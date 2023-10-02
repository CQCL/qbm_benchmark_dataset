"""
Benchmark a QBM on the Hamiltonian dataset
"""
import argparse
import numpy as np

from qbm_quimb import hamiltonians, data, training
from qbm_quimb.training import GibbsState, QBM

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
parser.add_argument("--qre", type=bool, default=True, help="True to output relative entropies")

args = parser.parse_args()

n_qubits = args.n
model_label = args.l
learning_rate = args.lr
epochs = args.e
eps = args.er
compute_qre = args.qre

########
# DATA #
########

# As an example, the Gibbs state of TF-Ising model (label=0)
# is taken to generate data (expectation values)
target_label = 0
target_ham_ops = hamiltonians.hamiltonian_operators(n_qubits, target_label)
target_params = np.array([4.0, 4.0])
target_beta = 2.0

target_state = GibbsState(target_ham_ops, target_params, target_beta)

# A list of operators in the model Hamiltonian
model_ham_ops = hamiltonians.hamiltonian_operators(n_qubits, model_label)
target_expects = target_state.compute_expectation(model_ham_ops)

#############
# QBM Model #
#############

initial_params = rng.normal(size=len(model_ham_ops))
qbm_state = QBM(model_ham_ops, initial_params)


################
# QBM Taininig #
################

target_eta = None
if compute_qre:
    target_eta = target_state

qbm_state, max_grads_hist, qre_hist = training.train_qbm(
    qbm_state,
    target_expects,
    learning_rate=learning_rate,
    epochs=epochs,
    eps=eps,
    compute_qre=compute_qre,
    target_eta=target_eta
)


print(f"target parameters: {target_params}")
print(f"trained parameters: {qbm_state.get_coeffs()}")
# print(f"Relative entropy: {qre_hist}")

# import matplotlib.pyplot as plt

# plt.plot(qre_hist[1:],'.')
# plt.xlabel("Epoch")
# plt.ylabel("Relative entropy")
# plt.show()
