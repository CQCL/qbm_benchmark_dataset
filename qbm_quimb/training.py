"""
Tools for training Quantum Boltzmann Machine with quantum relative entropy.
"""

import quimb as qu
import numpy as np


def qre(eta, h_qbm):
    """Quantum relative entropy

    Args:
        eta (Any): Target density matrix
        h_qbm (Any): Hamiltonian of the QBM
    """
    # h = qu.entropy(eta) # do not use because it has log2 inside
    evals = qu.eigvalsh(eta)
    h = np.sum(evals * np.log(evals))
    # use log base e all the way
    evals = qu.eigvalsh(h_qbm)
    z = np.sum(np.exp(evals))
    eta_stat = qu.expec(eta, h_qbm)
    return h-eta_stat+qu.log(z)

    

def compute_grads(
        h_i: list[np.ndarray], 
        h_i_exp: list[float], 
        rho: np.ndarray
    ) -> np.ndarray:
    """Compute gradients given a list of hamiltonian terms (operators)

    Args:
        h_i (list[np.ndarray]): A list of hamiltonian terms
        h_i_exp (list[float]): A list of hamiltonian expectation values
        rho (np.ndarray): The QBM density matrix

    Returns:
        np.ndarray: The array of the gradients
    """
    grads = []
    for h, eta_expect in zip(h_i, h_i_exp):
        rho_expect = qu.expec(rho,h)
        grads.append(rho_expect-eta_expect)
    return np.array(grads)

    

def training_qbm(
        h_i: list[np.ndarray], 
        h_i_exp: list[float],
        params: np.ndarray, 
        gamma: float = 0.2,
        epochs: int = 200, 
        eps: float = 1e-6
    ) -> list[float]:
    """Train QBM by computing gradients of relative entropy

    Args:
        hi (list[np.ndarray]): A list of hamiltonian terms
        h_i_exp (list[float]): A list of target hamiltonian expectation values
        params (np.ndarray): Parameters of QBM density matrix 
        epochs (int): The number of epochs in the training
        eps (float): Stop traninig when gradient gets smaller than eps

    Returns:
        list[float]]: Maximum absolute gradients
    """
    grad_hist = []

    for i in range(epochs):
        # create qbm hamiltonians
        qbm_tfim = 0
        for param, h in zip(params, h_i):
            qbm_tfim += param * h
        qbm_tfim = qbm_tfim.real
        # create qbm state
        rho = qu.thermal_state(qbm_tfim, -1.0)
        # grad and update
        grads = compute_grads(h_i, h_i_exp, rho)
        grad_hist.append(np.abs(grads))
        params = params - gamma * grads
        if np.max(grad_hist[-1]) < eps:
            break