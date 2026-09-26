"""LoRA adapters, effective deltas and parameter accounting.

All merging code in this package consumes the *effective* update
dW = scaling * B @ A of each adapter, never the factors themselves. The
factorisation (B, A) is only defined up to an invertible gauge R:
(B R, R^{-1} A) yields the same dW, so any method that depends on the factors
beyond dW is not a function of the adapted model.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from . import linalg as la
from .linalg import Matrix


@dataclass
class LoRAAdapter:
    """One LoRA adapter on one linear layer: dW = scaling * B @ A."""

    name: str
    B: Matrix  # (d_out, r)
    A: Matrix  # (r, d_in)
    scaling: float = 1.0  # e.g. lora_alpha / r in PEFT

    def __post_init__(self) -> None:
        d_out, r_b = la.shape(self.B)
        r_a, d_in = la.shape(self.A)
        if r_b != r_a:
            raise ValueError(f"adapter {self.name}: B has {r_b} cols but A has {r_a} rows")

    @property
    def rank_budget(self) -> int:
        return la.shape(self.A)[0]

    @property
    def d_out(self) -> int:
        return la.shape(self.B)[0]

    @property
    def d_in(self) -> int:
        return la.shape(self.A)[1]

    def effective_delta(self) -> Matrix:
        return la.scale(self.scaling, la.matmul(self.B, self.A))

    def gauge(self, r_mat: Matrix) -> "LoRAAdapter":
        """Return the equivalent adapter (B R, R^{-1} A)."""
        return LoRAAdapter(
            name=self.name,
            B=la.matmul(self.B, r_mat),
            A=la.matmul(la.inv(r_mat), self.A),
            scaling=self.scaling,
        )

    def n_params(self) -> int:
        return self.rank_budget * (self.d_out + self.d_in)


def random_gauge(r: int, cond: float, rng: random.Random) -> Matrix:
    """Random invertible r x r matrix with condition number `cond`."""
    q1 = la.random_orthogonal(r, rng)
    q2 = la.random_orthogonal(r, rng)
    sv = [cond ** (-(i / (r - 1))) if r > 1 else 1.0 for i in range(r)]
    sign = 1.0 if rng.random() < 0.5 else -1.0
    return la.scale(sign, la.matmul(la.matmul(q1, la.diag(sv)), q2))


def factorize_rank_k(m: Matrix, k: int) -> LoRAAdapter:
    """Balanced factorisation of a (rank <= k) matrix as a LoRA pair.

    Uses the truncated SVD m ~= U S V^T and sets B = U S^{1/2}, A = S^{1/2} V^T.
    If rank(m) > k this is the Eckart-Young truncation, so callers that need an
    exact factorisation must check the residual.
    """
    u, s, v = la.truncated_svd(m, k)
    rs = [x ** 0.5 for x in s]
    b = [[u[i][j] * rs[j] for j in range(len(s))] for i in range(len(u))]
    a = [[rs[j] * v[i][j] for i in range(len(v))] for j in range(len(s))]
    return LoRAAdapter(name="merged", B=b, A=a, scaling=1.0)


@dataclass
class ParamAccount:
    """Stored-parameter accounting for one layer."""

    d_out: int
    d_in: int
    input_adapter_params: int
    input_adapter_ranks: List[int]
    merged_rank: int
    merged_params_lowrank: int
    dense_params: int
    extra: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "d_out": self.d_out,
            "d_in": self.d_in,
            "input_adapter_ranks": self.input_adapter_ranks,
            "input_adapter_params_total": self.input_adapter_params,
            "merged_rank": self.merged_rank,
            "merged_params_if_stored_as_factors": self.merged_params_lowrank,
            "dense_delta_params": self.dense_params,
            "merged_over_input_ratio": (
                self.merged_params_lowrank / self.input_adapter_params
                if self.input_adapter_params else None
            ),
            **{f"extra_{k}": v for k, v in self.extra.items()},
        }


def param_account(adapters: Sequence[LoRAAdapter], merged_rank: int,
                  extra: Dict[str, int] | None = None) -> ParamAccount:
    d_out, d_in = adapters[0].d_out, adapters[0].d_in
    for ad in adapters:
        if (ad.d_out, ad.d_in) != (d_out, d_in):
            raise ValueError("adapters are not on the same layer shape")
    return ParamAccount(
        d_out=d_out,
        d_in=d_in,
        input_adapter_params=sum(ad.n_params() for ad in adapters),
        input_adapter_ranks=[ad.rank_budget for ad in adapters],
        merged_rank=merged_rank,
        merged_params_lowrank=merged_rank * (d_out + d_in),
        dense_params=d_out * d_in,
        extra=dict(extra or {}),
    )
