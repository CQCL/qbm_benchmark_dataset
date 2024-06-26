"""
Tools for training Quantum Boltzmann Machine with quantum relative entropy.
"""
import quimb as qu
import numpy as np
from typing import Optional

from qbm_quimb import hamiltonians


class GibbsState:
    """Parent class to represent a Gibbs state."""

    def __init__(
        self, ham_ops: list[qu.qarray], coeffs: np.ndarray[float], beta: float
    ):
        self.ham_ops = ham_ops
        self.coeffs = coeffs
        self.beta = beta

    def get_hamiltonian(self) -> qu.qarray:
        """Return total QBM Hamiltonian."""
        return hamiltonians.total_hamiltonian(self.ham_ops, self.coeffs)

    def get_coeffs(self) -> np.ndarray[float]:
        """Return a list of coefficients."""
        return self.coeffs

    def get_density_matrix(self) -> qu.qarray:
        """Return the density matrix of Gibbs state."""
        ham = self.get_hamiltonian()
        return qu.thermal_state(ham, self.beta)

    def compute_expectation(self, ops: list[qu.qarray]) -> list[float]:
        """Compute expectation values of a set of operators w.r.t the Gibbs state.

        Args:
            ops (float[qu.qarray]): A list of operators.

        Returns:
            list[float]: A list of expectation values.
        """
        rho = self.get_density_matrix()
        return [qu.expec(rho, op) for op in ops]


class QBM(GibbsState):
    """Represent a quantum Boltzmann machine exp(H) for a model Hamiltonian H."""

    def __init__(self, ham_ops: list[qu.qarray], coeffs: list[float]):
        """Initialize the Gibbs state corresponding to the QBM model.
        We use $\beta=-1$ to be consistent with the quimb.thermal_state function,
        so the QBM is $e^H/Z$.

        Args:
            ham_ops (list[qu.qarray]): List of operators in the Hamiltonian
            coeffs (list[float]): List of parameters, one for each operator in the Hamiltonian
        """  # noqa: E501
        super().__init__(ham_ops, coeffs, beta=-1.0)

    def compute_grads(
        self, target_expects: list[float], sigma: float = 0.0
    ) -> np.ndarray[float]:
        """Compute gradient of operator expectation values with the Gaussian shot noise of
        QBM and/or target expectation values.

        Args:
            target_expects (list[float]): A list of expectation values w.r.t. the target state.
            sigma (float): Standard deviation of the gradient due to the shot noise.

        Returns:
            np.ndarray[float]: An array of gradients.
        """  # noqa: E501
        qbm_expects = self.compute_expectation(self.ham_ops)
        noises = np.random.normal(loc=0, scale=sigma, size=len(self.ham_ops))
        grads = [
            qbm - targ + noise
            for (qbm, targ, noise) in zip(qbm_expects, target_expects, noises)
        ]
        return np.array(grads).real

    def update_params(self, grads: np.ndarray[float], learning_rate: float) -> None:
        """Update coefficients of Hamiltonian in QBM.

        Args:
            grads (np.ndarray[float]): A list of gradients of relative entropy.
            learning_rate (float): A learning rate.
        """
        self.coeffs = self.coeffs - learning_rate * grads

    def compute_qre(self, eta: qu.qarray, eta_evals: np.ndarray) -> float:
        """Compute quantum relative entropy between target (eta) and QBM states (rho),
        Tr[eta ln(eta) - eta ln(rho)].
        It uses the eigenvalues of the target state (eta) that have been
        pre-computed.

        Args:
            eta (qu.qarray): A target density martix.
            eta_evals (np.ndarray): The eigenvalues of the target state

        Returns:
            float: Quantum relative entropy.
        """
        # check if rank = 1, i.e., eta is a pure state
        if np.linalg.matrix_rank(eta.A) == 1:
            h = 0
        else:
            # Tr[eta ln(eta)] = sum(eta_ev * ln(eta_ev))
            h = np.sum(eta_evals * np.log(eta_evals))
        ham = self.get_hamiltonian()
        ham_evals = qu.eigvalsh(ham)
        # -Tr[eta ln(rho)] = -Tr[eta ln(e^H/Z)]
        # = -Tr[eta H + eta ln(Z)] = - Tr[eta H] - ln(Z)
        z = np.sum(np.exp(ham_evals))
        eta_stat = qu.expec(eta, ham)
        return h - np.real(eta_stat) + qu.log(z)


def train_qbm(
    qbm: QBM,
    target_expects: list[float],
    learning_rate: float = 0.2,
    epochs: int = 200,
    eps: float = 1e-6,
    sigma: float = 0.0,
    compute_qre: bool = False,
    target_eta: Optional[qu.qarray] = None,
    target_eta_ev: Optional[np.ndarray] = None,
) -> tuple[QBM, list[np.ndarray], list[float]]:
    """Training QBM given a list of target expectation values.

    Args:
        qbm (QBM): Quantum Bolzmann machine to be trained.
        target_expects (list[float]): A list of expectation values w.r.t the target state.
        learning_rate (float, optional): Learning rate. Defaults to 0.2.
        epochs (int, optional): Number of epochs. Defaults to 200.
        eps (float, optional): Threshold gradient below which the training loop is terminated. Defaults to 1e-6.
        compute_qre (bool, optional): Compute relative entropy if True. Defaults to False.
        target_eta (Optional[qu.qarray], optional): Target state used to compute relative entropy if compute_qre is True. Defaults to None.
        target_eta_ev (Optional[np.ndarray], optional): Target state eigenvalues for QRE if compute_qre is True. Defaults to None.

    Returns:
        QBM: The trained QBM.
        list[np.ndarray]: History of maxes of absolute values of gradients.
        list[float]]: History of relative entropies if compute_qre is True. Otherwise an empty list.
    """  # noqa: E501
    max_grad_hist = []
    qre_hist = []
    # initial QRE (always computed)
    qre_hist.append(qbm.compute_qre(target_eta, target_eta_ev))
    for _ in range(epochs):
        # grad and update
        grads = qbm.compute_grads(target_expects, sigma=sigma)
        max_grad_hist.append(np.max(np.abs(grads)))
        qbm.update_params(grads, learning_rate)
        # quantum relative entropy
        if compute_qre:
            qre_hist.append(qbm.compute_qre(target_eta, target_eta_ev))
        # stopping condition on gradients
        if max_grad_hist[-1] < eps:
            break
    if not compute_qre:
        # final QRE  (computed if not during training)
        qre_hist.append(qbm.compute_qre(target_eta, target_eta_ev))

    return qbm, max_grad_hist, qre_hist
