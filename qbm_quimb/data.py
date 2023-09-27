"""
Generate dataset that consists of expectation values of Hamiltonian terms 
with respect to density matrix.   
"""
import quimb as qu
from qbm_quimb import hamiltonians

    
def gibbs_expect(
        beta: float,
        coeffs: list[float],
        hamiltonian_ops: list[qu.qarray]
    ) -> tuple[list[float], qu.qarray]:
    """Create a density matrix for the Gibbs state and alculate thermal expectation values of operators

    Args:
        beta (float): Inverse temperature
        coeffs (list[float]): A list of coefficients
        hamiltonian_terms (list[qu.qarray]): A list of operators contained in the Hamitonian

    Returns:
        list[float]: A list of expactation values of Hamiltonian terms
        qu.qarray: Density matrix for the Gibbs state
    """
    hamiltonian = hamiltonians.total_hamiltonian(hamiltonian_ops, coeffs)
    rho = qu.thermal_state(hamiltonian, beta, precomp_func=False)
    hamiltonian_expectations = [qu.expec(rho, op) for op in hamiltonian_ops]
    return hamiltonian_expectations, rho