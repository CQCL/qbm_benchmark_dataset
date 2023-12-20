"""
Benchmark a QBM on the Hamiltonian dataset
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
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
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (0.01)")
parser.add_argument(
    "--e", type=int, default=1000, help="Number of traninig epochs (1000)"
)
parser.add_argument(
    "--er", type=float, default=1e-6, help="Error tolerance for gradients (1e-6)"
)
parser.add_argument(
    "--sn",
    type=float,
    default=0.0,
    help="Standard deviation of gaussian shot noise for computing gradients (0.0)",
)
parser.add_argument(
    "--qre",
    action="store_true",
    help="If we want to compute and output relative entropies",
)
parser.add_argument(
    "--pre_l", type=int, default=None, help="Label of QBM model for pretraining (None)"
)
parser.add_argument(
    "--pre_lr", type=float, default=0.01, help="Learning rate for pretraining (0.01)"
)
parser.add_argument(
    "--pre_e",
    type=int,
    default=300,
    help="Number of traninig epochs for pretraining (300)",
)

args = parser.parse_args()

n_qubits = args.n
model_label = args.l
depolarizing_noise = args.dn
learning_rate = args.lr
epochs = args.e
eps = args.er
shot_noise_sigma = args.sn
compute_qre = args.qre
pre_model_label = args.pre_l
pre_learning_rate = args.pre_lr
pre_epochs = args.pre_e

########
# DATA #
########

# As an example, the Gibbs state of TF-Ising model (label=0)
# is taken to generate data (expectation values)
target_label = 0
target_params = np.array([4.0, 4.0])
target_beta = 1.0

do_pretraining = pre_model_label is not None
if do_pretraining:
    stages = ["pretraining", "full-training"]
    _, model_ham_names_pre = hamiltonians.hamiltonian_operators(
        n_qubits, pre_model_label, return_names=True
    )
    _, model_ham_names_full = hamiltonians.hamiltonian_operators(
        n_qubits, model_label, return_names=True
    )
    assert len(set(model_ham_names_pre) - set(model_ham_names_full)) == 0
else:
    stages = ["full-training"]

for stage in stages:
    print(f"stage: {stage}")

    # A list of operators in the model Hamiltonian
    if stage == "pretraining":
        model_ham_ops, pre_model_ham_names = hamiltonians.hamiltonian_operators(
            n_qubits, pre_model_label, return_names=True
        )
    else:
        model_ham_ops, model_ham_names = hamiltonians.hamiltonian_operators(
            n_qubits, model_label, return_names=True
        )
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

    if do_pretraining and stage == "full-training":
        pre_op_map = {k: v for k, v in zip(pre_model_ham_names, qbm_state.get_coeffs())}
        # initialize parameters from pre-training
        initial_params = []
        for op in model_ham_names:
            try:
                if op in pre_model_ham_names:
                    initial_params.append(pre_op_map[op])
                else:
                    initial_params.append(0.0)
            except KeyError:
                initial_params.append(0)
    else:
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

    if stage == "pretraining":
        qbm_state, max_grads_hist, qre_hist = training.train_qbm(
            qbm=qbm_state,
            target_expects=target_expects,
            learning_rate=pre_learning_rate,
            epochs=pre_epochs,
            eps=eps,
            sigma=shot_noise_sigma,
            compute_qre=compute_qre,
            target_eta=target_eta,
        )
    else:
        qbm_state, max_grads_hist, qre_hist = training.train_qbm(
            qbm=qbm_state,
            target_expects=target_expects,
            learning_rate=learning_rate,
            epochs=epochs,
            eps=eps,
            sigma=shot_noise_sigma,
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

    fig_name = f"TFIM_beta{stringify(target_beta)}_q{n_qubits}_qbm{model_label}_e{epochs}_lr{stringify(learning_rate)}.png"  # noqa: E501
    if compute_qre:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(qre_hist[1:], "-")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Relative entropy")
        plt.savefig(f"data/figures/QRE_{fig_name}")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(max_grads_hist[1:], "-")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Max absolute value of gradients")
    plt.savefig(f"data/figures/MaxGrad_{fig_name}")
