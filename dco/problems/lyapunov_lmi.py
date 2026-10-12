"""Minimum-trace Lyapunov LMI SDPs."""

from jaxtyping import Float
import torch
from torch import Tensor

from ..cones import Cone, PositiveSemidefiniteCone
from ..linalg import svec, smat


def lyapunov_operator_matrix(A: Float[Tensor, "n n"]) -> Float[Tensor, "d d"]:
    """Build `M_A` such that `svec(A.T @ P + P @ A) = M_A @ svec(P)`."""
    n = A.shape[0]
    d = n * (n + 1) // 2
    M = torch.empty(d, d, dtype=A.dtype, device=A.device)

    for k in range(d):
        e_k = torch.zeros(d, dtype=A.dtype, device=A.device)
        e_k[k] = 1.0
        E = smat(e_k)
        M[:, k] = svec(A.mT @ E + E @ A)

    return M


def sample_schur_hurwitz_matrix(
    n: int,
    damping_min: float = 0.1,
    damping_max: float = 2.0,
    offdiag_scale: float = 1.0,
) -> Float[Tensor, "n n"]:
    """Sample a Hurwitz matrix using a random real Schur-form construction."""

    tensor_kwargs = {"dtype": torch.get_default_dtype()}
    Q, _ = torch.linalg.qr(torch.randn(n, n, **tensor_kwargs))

    T = offdiag_scale * torch.triu(torch.randn(n, n, **tensor_kwargs), diagonal=1)
    damping = damping_min + (damping_max - damping_min) * torch.rand(n, **tensor_kwargs)
    T = T + torch.diag(-damping)

    return Q @ T @ Q.mT


def sample_damped_oscillator_hurwitz_matrix(
    n: int,
    zeta_min: float = 0.05,
    zeta_max: float = 0.5,
    omega_min: float = 0.5,
    omega_max: float = 5.0,
    scalar_damping_min: float = 0.1,
    scalar_damping_max: float = 2.0,
) -> Float[Tensor, "n n"]:
    """Sample a Hurwitz matrix from damped oscillator blocks."""

    tensor_kwargs = {"dtype": torch.get_default_dtype()}
    blocks = []

    for _ in range(n // 2):
        zeta = zeta_min + (zeta_max - zeta_min) * torch.rand((), **tensor_kwargs)
        omega = omega_min + (omega_max - omega_min) * torch.rand((), **tensor_kwargs)
        blocks.append(
            torch.stack(
                [
                    torch.stack([-zeta * omega, omega]),
                    torch.stack([-omega, -zeta * omega]),
                ]
            )
        )

    if n % 2 == 1:
        damping = scalar_damping_min + (scalar_damping_max - scalar_damping_min) * torch.rand((), **tensor_kwargs)
        blocks.append(-damping.reshape(1, 1))

    A_block = torch.block_diag(*blocks)
    Q, _ = torch.linalg.qr(torch.randn(n, n, **tensor_kwargs))
    return Q @ A_block @ Q.mT


def sample_hurwitz_matrix(
    n: int,
    generator: str = "mixed",
    spectral_margin: float = 0.05,
    max_attempts: int = 100,
    schur_offdiag_scale: float = 1.0,
) -> Float[Tensor, "n n"]:
    """Sample a Hurwitz matrix with spectral abscissa below `-spectral_margin`.

    Args:
        schur_offdiag_scale: Scales the upper-triangle of the Schur factor `T`.
            Smaller values reduce the non-normality of the sampled `A`,
            thus affecting the conditioning of the Lyapunov operator `M_A`.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if generator not in {"mixed", "schur", "oscillator"}:
        raise ValueError("generator must be one of 'mixed', 'schur', or 'oscillator'.")

    for _ in range(max_attempts):
        generator_i = generator
        if generator_i == "mixed":
            generator_i = "schur" if torch.rand(()) < 0.5 else "oscillator"

        if generator_i == "schur":
            A = sample_schur_hurwitz_matrix(n, offdiag_scale=schur_offdiag_scale)
        else:
            A = sample_damped_oscillator_hurwitz_matrix(n)

        spectral_abscissa = torch.max(torch.linalg.eigvals(A).real)
        if spectral_abscissa < -spectral_margin:
            return A

    raise RuntimeError(f"Failed to sample a Hurwitz matrix with spectral margin {spectral_margin}.")


def generate_lyapunov_lmi_sdp(
    n: int = 5,
    generator: str = "mixed",
    spectral_margin: float = 0.05,
    layout: str = "csr",
    schur_offdiag_scale: float = 1.0,
) -> tuple[
    Float[Tensor, "d d"],
    Float[Tensor, "d"],
    Float[Tensor, "m d"],
    Float[Tensor, "m"],
    list[Cone],
]:
    """Generate a random minimum-trace Lyapunov LMI SDP."""

    A_sys = sample_hurwitz_matrix(
        n, generator=generator, spectral_margin=spectral_margin, schur_offdiag_scale=schur_offdiag_scale
    )

    n = A_sys.shape[0]
    d = n * (n + 1) // 2
    tensor_kwargs = {"dtype": A_sys.dtype, "device": A_sys.device}

    P = torch.zeros(d, d, **tensor_kwargs)
    eye_svec = svec(torch.eye(n, **tensor_kwargs))
    q = eye_svec.clone()

    M_A = lyapunov_operator_matrix(A_sys)
    A = torch.cat([M_A, -torch.eye(d, **tensor_kwargs)], dim=0)
    b = torch.cat([-eye_svec, -eye_svec], dim=0)

    cones = [
        PositiveSemidefiniteCone(torch.arange(0, d), gradient_method="dk"),
        PositiveSemidefiniteCone(torch.arange(d, 2 * d), gradient_method="dk"),
    ]

    if layout == "coo":
        P = P.to_sparse_coo()
        A = A.to_sparse_coo()
    elif layout == "csr":
        P = P.to_sparse_csr()
        A = A.to_sparse_csr()
    elif layout == "csc":
        P = P.to_sparse_csc()
        A = A.to_sparse_csc()
    elif layout == "dense":
        pass
    else:
        raise ValueError("layout must be one of 'coo', 'csr', 'csc', or 'dense'.")

    return P, q, A, b, cones
