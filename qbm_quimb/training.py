"""
Tools for training Quantum Boltzmann Machine with quantum relative entropy.
"""

from pyexpat import model
import quimb as qu
import numpy as np

from qbm_quimb import hamiltonians


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

    
def compute_expectations(
        rho: qu.qarray,
        operators: list[qu.qarray]
    ) -> list[float]:
    """Compute expectation values of operators w.r.t the density matrix rho

    Args:
        rho (qu.qarray): Density matrix
        operators (list[qu.qarray]): A list of operators

    Returns:
        list[float]: A list of expectation values
    """
    expectations = []
    for op in operators:
        expectations.append(qu.expec(rho, op))
    return expectations 


def compute_grads(
        qbm_expectations: list[float], 
        target_expectations: list[float], 
    ) -> np.ndarray:
    """Compute gradients given a list of hamiltonian terms (operators)

    Args:
        ham_terms (list[np.ndarray]): A list of hamiltonian terms
        ham_expectations (list[float])s A list of hamiltonian expectation values
        rho (qu.qarray): The QBM sdensity matrix

    Returns:
        np.ndarray: The array of the gradients
    """
    grads = [qbm-targ for (qbm,targ) in zip(qbm_expectations,target_expectations)]
    return np.array(grads)

    

def training_qbm(
        model_ham_ops: list[qu.qarray], 
        target_ham_expectations: list[float],
        target_eta: qu.qarray,
        initial_params: np.ndarray = None, 
        gamma: float = 0.2,
        epochs: int = 200, 
        eps: float = 1e-6
    ) -> tuple[np.ndarray, list[np.ndarray], list[float]]:
    """Train QBM by computing gradients of relative entropy

    Args:
        model_ham_ops (list[qu.qarray]): A list of operators in the model Hamiltonian
        target_ham_expectations (list[float]): A list of target hamiltonian expectation values
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
    max_grad_hist = []
    qre_hist = []
    
    if initial_params == None:
        params = np.zeros(len(model_ham_ops))
    else:
        params = initial_params

    for _ in range(epochs):
        # create qbm hamiltonian
        model_ham = hamiltonians.total_hamiltonian(model_ham_ops, params)
        model_ham = model_ham.real

        # create qbm state
        qbm_rho = qu.thermal_state(model_ham, 1.0)
        qbm_expectations = compute_expectations(qbm_rho, model_ham_ops)
        qre_hist.append(qre(target_eta, qbm_rho))

        # grad and update
        grads = compute_grads(qbm_expectations, target_ham_expectations)
        params = params - gamma * grads
        max_grad_hist.append(np.max(np.abs(grads)))
        if max_grad_hist[-1] < eps:
            break

    return params, max_grad_hist, qre_hist