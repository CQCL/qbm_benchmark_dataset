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


def annihilation_operator(n: int, index: int) -> qu.qarray:
    dims = (2,) * n

    def gen_term(i, k):
        return qu.ikron(qu.pauli(k), dims, [i])

    term = 1 / 2 * (gen_term(index, "X") + 1j * gen_term(index, "Y"))
    for i in range(index - 1, -1, -1):
        term = term @ gen_term(i, "Z")
    return term


def creation_operator(n: int, index: int) -> qu.qarray:
    return annihilation_operator(n, index).H


#########################
# Terms in Hamiltonians #
#########################


def hamiltonian_operators(
    n: int, label: int, cyclic: bool = False, return_names: bool = False
) -> list[qu.qarray] | tuple[list[qu.qarray], list[str]]:
    h_ops = []
    h_names = []

    match label:
        case 0:  # 1D transverse-field Ising model
            h_ops.append(h_two_sites(n, "Z"))
            h_ops.append(h_single_site(n, "X"))
            h_names.append("sum(Z_i_i+1)")
            h_names.append("sum(X_i)")

        case 1:  # 1D Heisenberg model
            h_ops.append(
                h_two_sites(n, "X") + h_two_sites(n, "Y") + h_two_sites(n, "Z")
            )
            h_ops.append(h_single_site(n, "Z"))
            h_names.append("sum(X_i_i+1 + Y_i_i+1 + Z_i_i+1)")
            h_names.append("sum(Z_i)")

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
            h_names.append("sum(X_i_i+1 + Y_i_i+1 + Z_i_i+1)")
            h_names.append("sum((-1)**i * (X_i_i+1 + Y_i_i+1 + Z_i_i+1))")

        case 3:  # J1-J2 model
            h_ops.append(
                h_two_sites(n, "X") + h_two_sites(n, "Y") + h_two_sites(n, "Z")
            )
            h_ops.append(
                h_two_sites_j2(n, "X") + h_two_sites_j2(n, "Y") + h_two_sites_j2(n, "Z")
            )
            h_names.append("sum(X_i_i+1 + Y_i_i+1 + Z_i_i+1)")
            h_names.append("sum(X_i_i+2 + Y_i_i+2 + Z_i_i+2)")

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
            h_names.append("hopping_term_1D")
            h_names.append("interaction_term_1D")

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
            h_names.append("hopping_term_2D")
            h_names.append("interaction_term_2D")

        case 6:  # Geometrically local (1D chain)
            dims = (2,) * n
            for k in ["X", "Y", "Z"]:
                for i in range(n - 1):
                    h_ops.append(qu.ikron(qu.pauli(k), dims, [i, i + 1]))
                    h_names.append(f"{k}_{i}_{i+1}")
                if cyclic:
                    h_ops.append(qu.ikron(qu.pauli(k), dims, [0, n - 1]))
                    h_names.append(f"{k}_{0}_{n-1}")
                for i in range(n):
                    h_ops.append(qu.ikron(qu.pauli(k), dims, [i]))
                    h_names.append(f"{k}_{i}")

        case 7:  # Geometrically local (2D lattice)
            assert (n**0.5).is_integer()
            sqrt_n = int(n**0.5)
            dims = (2,) * n
            for k in ["X", "Y", "Z"]:
                for i in range(n):
                    if i % sqrt_n != sqrt_n - 1:  # not the right end
                        h_ops.append(qu.ikron(qu.pauli(k), dims, [i, i + 1]))
                        h_names.append(f"{k}_{i}_{i+1}")
                    elif cyclic:
                        h_ops.append(qu.ikron(qu.pauli(k), dims, [i + 1 - sqrt_n, i]))
                        h_names.append(f"{k}_{i+1-sqrt_n}_{i}")
                    if i // sqrt_n != sqrt_n - 1:  # not the bottom end
                        h_ops.append(qu.ikron(qu.pauli(k), dims, [i, i + sqrt_n]))
                        h_names.append(f"{k}_{i}_{i+sqrt_n}")
                    elif cyclic:
                        h_ops.append(qu.ikron(qu.pauli(k), dims, [i % sqrt_n, i]))
                        h_names.append(f"{k}_{i % sqrt_n}_{i}")
                for i in range(n):
                    h_ops.append(qu.ikron(qu.pauli(k), dims, [i]))
                    h_names.append(f"{k}_{i}")

        case 8:  # Fully connected
            dims = (2,) * n
            for k in ["X", "Y", "Z"]:
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        h_ops.append(qu.ikron(qu.pauli(k), dims, [i, j]))
                        h_names.append(f"{k}_{i}_{j}")
                for i in range(n):
                    h_ops.append(qu.ikron(qu.pauli(k), dims, [i]))
                    h_names.append(f"{k}_{i}")

        case 9:  # Fully connected Classical
            dims = (2,) * n
            for i in range(n - 1):
                for j in range(i + 1, n):
                    h_ops.append(qu.ikron(qu.pauli("Z"), dims, [i, j]))
                    h_names.append(f"Z_{i}_{j}")
            for i in range(n):
                h_ops.append(qu.ikron(qu.pauli("Z"), dims, [i]))
                h_names.append(f"Z_{i}")

    if return_names:
        return h_ops, h_names
    else:
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
