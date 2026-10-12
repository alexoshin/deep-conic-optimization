"""Parametric families of cone programs.

Every generator returns `(P, q, A, b, cones)` for the problem
```
    min  0.5 x'Px + q'x,
    s.t. Ax + s = b, s in K.
```
"""

from .lovasz_theta import generate_lovasz_theta_sdp
from .lyapunov_lmi import generate_lyapunov_lmi_sdp
from .max_cut import generate_max_cut_sdp
from .robust_kf import generate_robust_kalman_filter_socp
from .robust_mpc import generate_robust_mpc_socp


GENERATORS = {
    "max_cut": generate_max_cut_sdp,
    "lovasz_theta": generate_lovasz_theta_sdp,
    "lyapunov_lmi": generate_lyapunov_lmi_sdp,
    "robust_kf": generate_robust_kalman_filter_socp,
    "robust_mpc": generate_robust_mpc_socp,
}
