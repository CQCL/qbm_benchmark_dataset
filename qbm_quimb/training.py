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
        ham_terms: list[qu.qarray], 
        ham_expectations: list[float], 
        rho: qu.qarray
    ) -> np.ndarray:
    """Compute gradients given a list of hamiltonian terms (operators)

    Args:
        ham_terms (list[np.ndarray]): A list of hamiltonian terms
        ham_expectations (list[float]): A list of hamiltonian expectation values
        rho (qu.qarray): The QBM density matrix

    Returns:
        np.ndarray: The array of the gradients
    """
    grads = []
    for h, eta_expect in zip(ham_terms, ham_expectations):
        rho_expect = qu.expec(rho,h)
        grads.append(rho_expect-eta_expect)
    return np.array(grads)

    

def training_qbm(
        ham_terms: list[qu.qarray], 
        ham_expectations: list[float],
        target_eta: qu.qarray,
        params: np.ndarray = None, 
        gamma: float = 0.2,
        epochs: int = 200, 
        eps: float = 1e-6
    ) -> tuple[np.ndarray, list[np.ndarray], list[float]]:
    """Train QBM by computing gradients of relative entropy

    Args:
        ham_terms (list[qu.qarray]): A list of hamiltonian terms
        ham_expectations (list[float]): A list of target hamiltonian expectation values
        target_eta (qu.qarray): Target density matrix
        params (np.ndarray): Parameters of QBM density matrix 
        gamma (float): Learning rate
        epochs (int): The number of epochs in the training
        eps (float): Stop traninig when gradient gets smaller than eps

    Returns:
        np.ndarray: Parameters of the trained QBM desity matrix
        list[np.ndarray]: Maximum absolute gradients
        list[float]: A list of relative entropies
    """
    grad_hist = []
    qre_hist = []
    
    if params == None:
        new_params = np.zeros(len(ham_terms))
    else:
        new_params = params

    for i in range(epochs):
        # create qbm hamiltonians
        qbm_tfim = 0.0
        for param, h in zip(new_params, ham_terms):
            qbm_tfim += param * h
        qbm_tfim = qbm_tfim.real
        qre_hist.append(qre(target_eta, qbm_tfim))
        # create qbm state
        rho = qu.thermal_state(qbm_tfim, -1.0)
        # grad and update
        grads = compute_grads(ham_terms, ham_expectations, rho)
        grad_hist.append(np.abs(grads))
        new_params = new_params - gamma * grads
        if np.max(grad_hist[-1]) < eps:
            break

    return new_params, grad_hist, qre_hist