"""
Generate dataset that consists of expectation values of Hamiltonian terms
with respect to density matrix.
"""
import pickle
import numpy as np
import quimb as qu
from qbm_quimb import hamiltonians
from qbm_quimb.training import GibbsState
from typing import Optional


def gibbs_expect(
    beta: float, coeffs: list[float], hamiltonian_ops: list[qu.qarray]
) -> tuple[list[float], qu.qarray]:
    """Create a density matrix for the Gibbs state and calculate thermal expectation values of operators

    Args:
        beta (float): Inverse temperature
        coeffs (list[float]): A list of coefficients
        hamiltonian_terms (list[qu.qarray]): A list of operators contained in the Hamitonian

    Returns:
        list[float]: A list of expactation values of Hamiltonian terms
        qu.qarray: Density matrix for the Gibbs state
    """  # noqa: E501
    hamiltonian = hamiltonians.total_hamiltonian(hamiltonian_ops, coeffs)
    rho = qu.thermal_state(hamiltonian, beta, precomp_func=False)
    hamiltonian_expectations = [qu.expec(rho, op) for op in hamiltonian_ops]
    return hamiltonian_expectations, rho


def generate_data(
    n_qubits: int,
    target_label: int,
    target_params: np.ndarray[float],
    target_beta: float,
    hamiltonian_ops: list[qu.qarray],
    depolarizing_noise: float = 0.0,
    file_path: Optional[str] = None,
) -> tuple[list[float], qu.qarray]:
    """Create a density matrix for the Gibbs state and calculate thermal expectation values of operators

    Args:
        n_qubits (int): The number of qubits
        target_label (int): Hamiltonian label for target
        target_params (np.ndarray[float]): Parameters of the target Hamiltonian
        target_beta (float): Inverse temperature
        hamiltonian_ops (list[qu.qarray]): A list of operators contained in the Hamitonian
        depolarizing_noise (float): Intensity of depolarizing noise
        file_path (Optional[str]): Output file path

    Returns:
        list[float]: A list of expactation values of Hamiltonian terms
        qu.qarray: Density matrix for the target state (could be noisy)
    """
    target_ham_ops = hamiltonians.hamiltonian_operators(n_qubits, target_label)
    target_state = GibbsState(target_ham_ops, target_params, target_beta)
    eta = (
        1 - depolarizing_noise
    ) * target_state.get_density_matrix() + depolarizing_noise * qu.qarray(
        np.eye(2**n_qubits), dtype=complex
    ) / 2**n_qubits
    target_expects = [qu.expec(eta, op) for op in hamiltonian_ops]
    if file_path:
        with open(file_path, mode="wb") as f:
            pickle.dump((target_expects, eta), f)

    return target_expects, eta
