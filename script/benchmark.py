"""
Benchmark a QBM on the Hamiltonian dataset
"""
import argparse
import numpy as np
from time import time
from qbm_quimb import hamiltonians, data, training
from qbm_quimb.training import QBM


def stringify(p: float):
    return str(p).replace(".", "-")


##########
# CONFIG #
##########

rng = np.random.default_rng(seed=1)

# CLI arguments:
parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=8, help="Number of qubits (8)")
parser.add_argument("--l", type=int, default=0, help="Label of QBM model (0)")
parser.add_argument(
    "--dn", type=float, default=0.0, help="Intensity of depolarizing noise (0.0)"
)
parser.add_argument("--lr", type=float, default=0.2, help="Learning rate (0.2)")
parser.add_argument(
    "--e", type=int, default=200, help="Number of traninig epochs (200)"
)
parser.add_argument(
    "--er", type=int, default=1e-6, help="Error tolerance for gradients (1e-6)"
)
parser.add_argument(
    "--qre",
    action="store_true",
    help="If we want to compute and output relative entropies",
)

args = parser.parse_args()

n_qubits = args.n
model_label = args.l
depolarizing_noise = args.dn
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
target_params = np.array([4.0, 4.0])
target_beta = 1.0

# A list of operators in the model Hamiltonian
model_ham_ops = hamiltonians.hamiltonian_operators(n_qubits, model_label)
target_expects, target_state = data.generate_data(
    n_qubits,
    target_label,
    target_params,
    target_beta,
    model_ham_ops,
    depolarizing_noise,
)

#############
# QBM Model #
#############

initial_params = rng.normal(size=len(model_ham_ops))
qbm_state = QBM(model_ham_ops, initial_params)
print(f"Initial parameters: {qbm_state.get_coeffs()}")
print(f"Target parameters: {target_params}")
print(f"Target beta: {target_beta}")

################
# QBM Taininig #
################

start_time = time()
print("Start training...")
target_eta = None
if compute_qre:
    target_eta = target_state

qbm_state, max_grads_hist, qre_hist = training.train_qbm(
    qbm=qbm_state,
    target_expects=target_expects,
    learning_rate=learning_rate,
    epochs=epochs,
    eps=eps,
    compute_qre=compute_qre,
    target_eta=target_eta,
)
end_time = time()
print(f"Training took {(end_time-start_time):.2f}s to run")
print(f"Trained parameters: {qbm_state.get_coeffs()}")
print(f"Max. gradients: {max_grads_hist[-1]}")
if compute_qre:
    print(f"Initial relative entropy: {qre_hist[0]}")
    print(f"Trained relative entropy: {qre_hist[-1]}")

import matplotlib.pyplot as plt

fig_name = f"TFIM_beta{stringify(target_beta)}_q{n_qubits}_qbm{model_label}_e{epochs}_lr{stringify(learning_rate)}.png"
if compute_qre:
    plt.plot(qre_hist[1:], "-")
    plt.xlabel("Epoch")
    plt.ylabel("Relative entropy")
    plt.savefig(f"data/figures/QRE_{fig_name}")
plt.plot(max_grads_hist[1:], "-")
plt.xlabel("Epoch")
plt.ylabel("Max absolute value of gradients")
plt.savefig(f"data/figures/MaxGrad_{fig_name}")
