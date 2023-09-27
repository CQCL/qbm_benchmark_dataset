"""
Generate a list of operators contained in each Hamiltonian.
"""
import quimb as qu
import numpy as np


###########################
# 1- and 2-spin operators #
###########################


def h_single_site(n, k):
    dims = (2,) * n

    def gen_term(i: int):
        return qu.ikron(qu.pauli(k), dims, [i])

    return sum(map(gen_term, range(0, n)))


def h_two_sites(n, k):
    dims = (2,) * n

    def gen_term(i):
        return qu.ikron(qu.pauli(k), dims, [i, i + 1])

    return sum(map(gen_term, range(0, n - 1)))


def h_two_sites_j2(n, k):
    dims = (2,) * n

    def gen_term(i):
        return qu.ikron(qu.pauli(k), dims, [i, i + 2])

    return sum(map(gen_term, range(0, n - 2)))


################################
# Jordan-Wigner transformation #
################################


def annihilation_operator(n, index):
    dims = (2,) * n

    def gen_term(i, k):
        return qu.ikron(qu.pauli(k), dims, [i])

    term = 1 / 2 * (gen_term(index, "X") + 1j * gen_term(index, "Y"))
    for i in range(index - 1, -1, -1):
        term = term @ gen_term(i, "Z")
    return term


def creation_operator(n, index):
    return annihilation_operator(n, index).H


#########################
# Terms in Hamiltonians #
#########################


def hamiltonian_operators(n: int, label: int) -> list[qu.qarray]:
    h_ops = []

    match label:
        case 0:  # 1D transverse-field Ising model
            h_ops.append(h_two_sites(n, "Z"))
            h_ops.append(h_single_site(n, "X"))

        case 1:  # 1D Heisenberg model
            h_ops.append(
                h_two_sites(n, "X") + h_two_sites(n, "Y") + h_two_sites(n, "Z")
            )
            h_ops.append(h_single_site(n, "Z"))

        case 2:  # Su-Schrieffer-Heeger (SSH) model
            h_ops.append(
                h_two_sites(n, "X") + h_two_sites(n, "Y") + h_two_sites(n, "Z")
            )
            op = qu.qarray(np.zeros((2**n, 2**n), dtype=complex))
            dims = (2,) * n
            for k in ["X", "Y", "Z"]:
                for i in range(n - 1):
                    op += (-1) ** i * qu.ikron(qu.pauli(k), dims, [i, i + 1])
            h_ops.append(op)

        case 3:  # J1-J2 model
            h_ops.append(
                h_two_sites(n, "X") + h_two_sites(n, "Y") + h_two_sites(n, "Z")
            )
            h_ops.append(
                h_two_sites_j2(n, "X") + h_two_sites_j2(n, "Y") + h_two_sites_j2(n, "Z")
            )

        case 4:  # 1D Hubbard model
            assert n % 2 == 0
            # hopping term
            hopping_term = qu.qarray(np.zeros((2**n, 2**n), dtype=complex))
            for j in range(n // 2 - 1):
                for sigma in range(2):
                    hopping_term -= creation_operator(
                        n, 2 * j + sigma
                    ) @ annihilation_operator(
                        n, 2 * (j + 1) + sigma
                    ) + creation_operator(
                        n, 2 * (j + 1) + sigma
                    ) @ annihilation_operator(
                        n, 2 * j + sigma
                    )
            h_ops.append(hopping_term)
            # interaction term
            interaction_term = qu.qarray(np.zeros((2**n, 2**n), dtype=complex))
            for j in range(n // 2):
                interaction_term += (
                    creation_operator(n, 2 * j) @ annihilation_operator(n, 2 * j)
                    - 1 / 2 * qu.identity(2**n)
                ) @ (
                    creation_operator(n, 2 * j + 1)
                    @ annihilation_operator(n, 2 * j + 1)
                    - 1 / 2 * qu.identity(2**n)
                )
            h_ops.append(interaction_term)

        case 5:  # 2D Hubbard model
            assert n >= 8 and n % 4 == 0
            # hopping term
            hopping_term = qu.qarray(np.zeros((2**n, 2**n), dtype=complex))
            for sigma in range(2):
                for jx in range(n // 4 - 1):
                    for jy in range(2):
                        j = n // 4 * jy + jx
                        hopping_term -= creation_operator(
                            n, 2 * j + sigma
                        ) @ annihilation_operator(
                            n, 2 * (j + 1) + sigma
                        ) + creation_operator(
                            n, 2 * (j + 1) + sigma
                        ) @ annihilation_operator(
                            n, 2 * j + sigma
                        )
                for jx in range(n // 4):
                    hopping_term -= creation_operator(
                        n, 2 * jx + sigma
                    ) @ annihilation_operator(
                        n, 2 * (n // 4 + jx) + sigma
                    ) + creation_operator(
                        n, 2 * (n // 4 + jx) + sigma
                    ) @ annihilation_operator(
                        n, 2 * jx + sigma
                    )
            h_ops.append(hopping_term)
            # interaction term
            interaction_term = qu.qarray(np.zeros((2**n, 2**n), dtype=complex))
            for jx in range(n // 4):
                for jy in range(2):
                    j = n // 4 * jy + jx
                    interaction_term += (
                        creation_operator(n, 2 * j) @ annihilation_operator(n, 2 * j)
                        - 1 / 2 * qu.identity(2**n)
                    ) @ (
                        creation_operator(n, 2 * j + 1)
                        @ annihilation_operator(n, 2 * j + 1)
                        - 1 / 2 * qu.identity(2**n)
                    )
            h_ops.append(interaction_term)

    return h_ops


def total_hamiltonian(
    hamiltonian_operators: list[qu.qarray], coeffs: list[float]
) -> qu.qarray:
    """Create a Hamiltonian operator

    Args:
        hamiltonian_operators (list[qu.qarray]): Operators in the Hamiltonian
        coeffs (list[float]): A list of coefficients of the operators

    Returns:
        qu.qarray: A Hamiltonian operator
    """
    assert len(hamiltonian_operators) == len(
        coeffs
    ), "Mismatch between the lengths of hamiltonian_terms and coeffs!"
    hamiltonian_terms = []
    for op, coeff in zip(hamiltonian_operators, coeffs):
        hamiltonian_terms.append(coeff * op)
    return sum(hamiltonian_terms)
