"""
Generate dataset that consists of expectation values of Hamiltonian terms 
with respect to density matrix.   
"""
import quimb as qu

def tfim_gibbs_expect(
        n: int,
        jz: float,
        bx: float,
        beta: float,
        hamiltonian_terms: list[qu.qarray]
    ) -> tuple[qu.qarray, list[float]]:
    """_summary_

    Args:
        n (int): Number of qubits
        jz (float): 
        bx (float): 
        beta (float): 
        hamiltonian_terms (list[qu.qarray]): A list of Hamitonian terms

    Returns:
        list[float]: A list of expactation values of Hamiltonian terms
    """
    tfim = qu.ham_ising(n, jz, bx)
    eta_tfim = qu.thermal_state(tfim, beta, precomp_func=False)
    hamiltonian_expectations = [qu.expec(eta_tfim, h) for h in hamiltonian_terms]
    return eta_tfim, hamiltonian_expectations