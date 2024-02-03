"""
Benchmark a QBM on the Hamiltonian dataset
"""

import argparse
import os
import numpy as np
import quimb as qu
import matplotlib.pyplot as plt
from time import time
from qbm_quimb import hamiltonians, data, training
from qbm_quimb.training import QBM


def stringify(p: float):
    return str(p).replace(".", "-")


##########
# CONFIG #
##########
# CLI arguments:
parser = argparse.ArgumentParser(
    description="Train a QBM model to represent a target Gibbs state"
)
parser.add_argument("--n", type=int, default=4, help="Number of qubits (4)")
parser.add_argument("--t", type=int, default=0, help="Label of target model (0)")
parser.add_argument(
    "--b", type=float, default=1.0, help="Inverse temperature of target state (1.0)"
)
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
    help="If we want to compute and output relative entropies during training",
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
parser.add_argument(
    "--seed",
    type=int,
    default=1,
    help="Seed for PRNG (1)",
)
parser.add_argument(
    "--output",
    type=str,
    default="data/",
    help="Output for data and figures (data/)",
)

args = parser.parse_args()

n_qubits = args.n
target_label = args.t
target_beta = args.b
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
output_path = args.output
os.makedirs(f"{output_path}/figures", exist_ok=True)
os.makedirs(f"{output_path}/histories", exist_ok=True)
os.makedirs(f"{output_path}/results", exist_ok=True)

if n_qubits > 10:
    print("It will be expensive to compute the QRE at each step of the training.")
    print("We only compute it at the beginning and at the end of training")
    compute_qre = False

# DATA #
########
rng = np.random.default_rng(seed=args.seed)

# Initialize the Gibbs state of the target
# by choosing some random parameters
# as the coefficients of each hamiltonian operator
target_ham_ops = hamiltonians.hamiltonian_operators(n_qubits, target_label)
target_params = rng.normal(size=len(target_ham_ops))
print(f"Target Hamiltonian label {target_label}")
print(f"Model Hamiltonian label {model_label}")
if target_label == model_label:
    print(" -- no model mismatch")
else:
    print(" -- model mismatch")

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
        pre_op_map = {
            k: v
            for k, v in zip(pre_model_ham_names, qbm_state.get_coeffs())  # noqa: F821
        }
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
    target_eta = target_state
    target_eta_ev = qu.eigvalsh(target_eta).clip(1e-300)

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
            target_eta_ev=target_eta_ev,
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
            target_eta_ev=target_eta_ev,
        )
    end_time = time()

    exp_name = f"t{target_label}_beta{stringify(target_beta)}_q{n_qubits}_qbm{model_label}_e{epochs}_lr{stringify(learning_rate)}_sn{stringify(shot_noise_sigma)}_dn{stringify(depolarizing_noise)}"  # noqa: E501
    if do_pretraining:
        exp_name += (
            f"_pre-qbm{pre_model_label}"
            + f"_pre-e{pre_epochs}"
            + f"_pre-lr{stringify(pre_learning_rate)}_"
        )
        if stage == "pretraining":
            exp_name += "pre"
        else:
            exp_name += "full"

    print(f"Training took {(end_time-start_time):.2f}s to run")
    print(f"Trained parameters: {qbm_state.get_coeffs()}")
    print(f"Max. gradients: {max_grads_hist[-1]}")
    np.save(f"{output_path}/results/Time_{exp_name}.npy", end_time - start_time)
    np.save(f"{output_path}/results/ParamsTrue_{exp_name}.npy", target_params)
    np.save(f"{output_path}/results/Params_{exp_name}.npy", qbm_state.get_coeffs())
    np.save(f"{output_path}/histories/MaxGrad_{exp_name}.npy", max_grads_hist)
    # Plot gradients: difference in expectation values
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(max_grads_hist[1:], "-")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Max absolute value of gradients")
    ax.set_yscale("log")
    plt.savefig(f"{output_path}/figures/MaxGrad_{exp_name}.png")
    # Plot QRE: training loss function
    print(f"Initial relative entropy: {qre_hist[0]}")
    print(f"Trained relative entropy: {qre_hist[-1]}")
    np.save(f"{output_path}/histories/QRE_{exp_name}.npy", qre_hist)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(qre_hist[1:], "-")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Relative entropy")
    ax.set_yscale("log")
    plt.savefig(f"{output_path}/figures/QRE_{exp_name}.png")
