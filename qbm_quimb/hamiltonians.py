import quimb as qu
from typing import List


def h_single_site(n: int, k: str):
    assert k in ["X", "Y", "Z"]
    dims = (2,) * n

    def gen_term(i):
        return qu.ikron(qu.pauli(k), dims, [i])

    return sum(map(gen_term, range(0, n)))


def h_two_sites_1(n: int, k: str):
    assert k in ["X", "Y", "Z"]
    dims = (2,) * n

    def gen_term(i):
        return qu.ikron(qu.pauli(k), dims, [i, i + 1])

    return sum(map(gen_term, range(0, n - 1)))


def h_two_sites_2(n: int, k: str):
    assert k in ["X", "Y", "Z"]
    dims = (2,) * n

    def gen_term(i):
        return qu.ikron(qu.pauli(k), dims, [i, i + 2])

    return sum(map(gen_term, range(0, n - 2)))


def Hamiltonian_terms(n: int, label: int) -> List:
    h_terms = []

    if label == 0:
        h_terms.append(h_two_sites_1(n, "Z"))
        h_terms.append(h_single_site(n, "X"))
    if label == 1:
        h_terms.append(
            h_two_sites_1(n, "X") + h_two_sites_1(n, "Y") + h_two_sites_1(n, "Z")
        )
        h_terms.append(h_single_site(n, "Z"))
    if label == 2:
        pass  # to be implemented
    if label == 3:
        h_terms.append(
            h_two_sites_1(n, "X") + h_two_sites_1(n, "Y") + h_two_sites_1(n, "Z")
        )
        h_terms.append(
            h_two_sites_2(n, "X") + h_two_sites_2(n, "Y") + h_two_sites_2(n, "Z")
        )
    if label == 4:
        pass  # to be implemented
    if label == 5:
        pass  # to be implemented

    return h_terms
