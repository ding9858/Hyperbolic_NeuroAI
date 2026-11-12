"""
Poincare-ball geometry for inference-time hyperbolic latent drifting.

The diffusion U-Net and VAE remain Euclidean and frozen.
Only the scalar drift applied after a DDIM transition is mapped to
and from a local Poincare ball.
"""

from __future__ import annotations

import math

import torch


EPS = 1e-8


def _validate_curvature(curvature: float) -> None:
    if not math.isfinite(curvature) or curvature < 0.0:
        raise ValueError("Curvature must be a finite non-negative scalar.")


def _safe_norm(tensor: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(
        tensor,
        dim=-1,
        keepdim=True,
    ).clamp_min(EPS)


def poincare_radius(curvature: float) -> float:
    _validate_curvature(curvature)

    if curvature == 0.0:
        return math.inf

    return 1.0 / math.sqrt(curvature)


def project_to_poincare_ball(
    point: torch.Tensor,
    curvature: float,
    projection_eps: float = 1e-5,
) -> torch.Tensor:
    """
    Project points to the open Poincare ball with curvature -curvature.
    """

    _validate_curvature(curvature)

    if curvature == 0.0:
        return point

    if not 0.0 < projection_eps < 1.0:
        raise ValueError("projection_eps must lie strictly between 0 and 1.")

    maximum_norm = (
        (1.0 - projection_eps)
        / math.sqrt(curvature)
    )

    norm = _safe_norm(point)

    scale = torch.clamp(
        point.new_tensor(maximum_norm) / norm,
        max=1.0,
    )

    return point * scale


def poincare_expmap0(
    tangent_vector: torch.Tensor,
    curvature: float,
    projection_eps: float = 1e-5,
) -> torch.Tensor:
    """
    Exponential map from the origin to the Poincare ball.
    """

    _validate_curvature(curvature)

    if curvature == 0.0:
        return tangent_vector

    sqrt_curvature = math.sqrt(curvature)
    tangent_norm = _safe_norm(tangent_vector)
    argument = sqrt_curvature * tangent_norm

    scale = torch.tanh(argument) / argument

    point = scale * tangent_vector

    return project_to_poincare_ball(
        point,
        curvature=curvature,
        projection_eps=projection_eps,
    )


def poincare_logmap0(
    point: torch.Tensor,
    curvature: float,
    projection_eps: float = 1e-5,
) -> torch.Tensor:
    """
    Logarithmic map from the Poincare ball to the tangent space at the origin.
    """

    _validate_curvature(curvature)

    if curvature == 0.0:
        return point

    point = project_to_poincare_ball(
        point,
        curvature=curvature,
        projection_eps=projection_eps,
    )

    sqrt_curvature = math.sqrt(curvature)
    point_norm = _safe_norm(point)

    argument = torch.clamp(
        sqrt_curvature * point_norm,
        max=1.0 - projection_eps,
    )

    scale = torch.atanh(argument) / argument.clamp_min(EPS)

    return scale * point


def poincare_conformal_factor(
    point: torch.Tensor,
    curvature: float,
    projection_eps: float = 1e-5,
) -> torch.Tensor:
    """
    Conformal metric factor lambda_x = 2 / (1 - c ||x||^2).
    """

    _validate_curvature(curvature)

    if curvature == 0.0:
        return torch.ones_like(point[..., :1])

    point = project_to_poincare_ball(
        point,
        curvature=curvature,
        projection_eps=projection_eps,
    )

    squared_norm = torch.sum(
        point.square(),
        dim=-1,
        keepdim=True,
    )

    denominator = (
        1.0 - curvature * squared_norm
    ).clamp_min(EPS)

    return 2.0 / denominator


def poincare_mobius_add(
    first: torch.Tensor,
    second: torch.Tensor,
    curvature: float,
    projection_eps: float = 1e-5,
) -> torch.Tensor:
    """
    Mobius addition on the Poincare ball with curvature -curvature.
    """

    _validate_curvature(curvature)

    if curvature == 0.0:
        return first + second

    first = project_to_poincare_ball(
        first,
        curvature=curvature,
        projection_eps=projection_eps,
    )
    second = project_to_poincare_ball(
        second,
        curvature=curvature,
        projection_eps=projection_eps,
    )

    first_squared_norm = torch.sum(
        first.square(),
        dim=-1,
        keepdim=True,
    )
    second_squared_norm = torch.sum(
        second.square(),
        dim=-1,
        keepdim=True,
    )
    inner_product = torch.sum(
        first * second,
        dim=-1,
        keepdim=True,
    )

    numerator = (
        (1.0 + 2.0 * curvature * inner_product
         + curvature * second_squared_norm) * first
        + (1.0 - curvature * first_squared_norm) * second
    )

    denominator = (
        1.0
        + 2.0 * curvature * inner_product
        + curvature**2 * first_squared_norm * second_squared_norm
    ).clamp_min(EPS)

    return project_to_poincare_ball(
        numerator / denominator,
        curvature=curvature,
        projection_eps=projection_eps,
    )


def poincare_expmap(
    base_point: torch.Tensor,
    tangent_vector: torch.Tensor,
    curvature: float,
    projection_eps: float = 1e-5,
) -> torch.Tensor:
    """
    Exponential map at an arbitrary Poincare-ball point.
    """

    _validate_curvature(curvature)

    if curvature == 0.0:
        return base_point + tangent_vector

    base_point = project_to_poincare_ball(
        base_point,
        curvature=curvature,
        projection_eps=projection_eps,
    )

    sqrt_curvature = math.sqrt(curvature)
    tangent_norm = _safe_norm(tangent_vector)

    conformal_factor = poincare_conformal_factor(
        base_point,
        curvature=curvature,
        projection_eps=projection_eps,
    )

    argument = (
        sqrt_curvature
        * conformal_factor
        * tangent_norm
        / 2.0
    )

    mobius_tangent = (
        torch.tanh(argument)
        * tangent_vector
        / (sqrt_curvature * tangent_norm)
    )

    return poincare_mobius_add(
        base_point,
        mobius_tangent,
        curvature=curvature,
        projection_eps=projection_eps,
    )


def parallel_transport_from_origin(
    point: torch.Tensor,
    tangent_vector_at_origin: torch.Tensor,
    curvature: float,
    projection_eps: float = 1e-5,
) -> torch.Tensor:
    """
    Parallel transport from the origin to a Poincare-ball point.
    """

    _validate_curvature(curvature)

    if curvature == 0.0:
        return tangent_vector_at_origin

    conformal_factor = poincare_conformal_factor(
        point,
        curvature=curvature,
        projection_eps=projection_eps,
    )

    return (2.0 / conformal_factor) * tangent_vector_at_origin


def latent_to_local_blocks(latent: torch.Tensor) -> torch.Tensor:
    """
    Convert [batch, channels, height, width] to [batch, height, width, channels].
    """

    if latent.ndim != 4:
        raise ValueError(
            "Expected latent shape [batch, channels, height, width]."
        )

    return latent.permute(0, 2, 3, 1).contiguous()


def local_blocks_to_latent(local_blocks: torch.Tensor) -> torch.Tensor:
    """
    Convert [batch, height, width, channels] to [batch, channels, height, width].
    """

    if local_blocks.ndim != 4:
        raise ValueError(
            "Expected local block shape [batch, height, width, channels]."
        )

    return local_blocks.permute(0, 3, 1, 2).contiguous()


def make_euclidean_scalar_drift(delta: float):
    """
    Euclidean LD-style reverse update:
    z_(t-1) = base_DDIM_update + delta.
    """

    delta = float(delta)

    def transition_modifier(
        x_t,
        base_x_prev,
        predicted_epsilon,
        predicted_x0,
        timestep,
        reverse_step,
    ):
        return base_x_prev + base_x_prev.new_tensor(delta)

    return transition_modifier


def make_hyperbolic_scalar_drift(
    delta: float,
    curvature: float,
    coordinate_scale: float = 0.2,
    projection_eps: float = 1e-5,
):
    """
    Hyperbolic counterpart of Euclidean scalar latent drift.

    The base DDIM transition stays unchanged. Each local six-channel vector
    of base_x_prev is mapped to a Poincare ball, receives a parallel-transported
    scalar drift through the exponential map, and is mapped back.

    delta = 0 returns base_x_prev exactly.
    curvature = 0 returns the Euclidean scalar drift exactly.
    """

    delta = float(delta)
    curvature = float(curvature)
    coordinate_scale = float(coordinate_scale)

    _validate_curvature(curvature)

    if coordinate_scale <= 0.0:
        raise ValueError("coordinate_scale must be positive.")

    if delta == 0.0:
        def zero_drift_modifier(
            x_t,
            base_x_prev,
            predicted_epsilon,
            predicted_x0,
            timestep,
            reverse_step,
        ):
            return base_x_prev

        return zero_drift_modifier

    if curvature == 0.0:
        return make_euclidean_scalar_drift(delta)

    def transition_modifier(
        x_t,
        base_x_prev,
        predicted_epsilon,
        predicted_x0,
        timestep,
        reverse_step,
    ):
        base_blocks = latent_to_local_blocks(base_x_prev)

        hyperbolic_point = poincare_expmap0(
            coordinate_scale * base_blocks,
            curvature=curvature,
            projection_eps=projection_eps,
        )

        drift_at_origin = torch.full_like(
            base_blocks,
            fill_value=coordinate_scale * delta,
        )

        drift_at_point = parallel_transport_from_origin(
            hyperbolic_point,
            drift_at_origin,
            curvature=curvature,
            projection_eps=projection_eps,
        )

        shifted_hyperbolic_point = poincare_expmap(
            hyperbolic_point,
            drift_at_point,
            curvature=curvature,
            projection_eps=projection_eps,
        )

        shifted_blocks = (
            poincare_logmap0(
                shifted_hyperbolic_point,
                curvature=curvature,
                projection_eps=projection_eps,
            )
            / coordinate_scale
        )

        return local_blocks_to_latent(shifted_blocks)

    return transition_modifier


def run_geometry_sanity_checks() -> None:
    """
    Verify algebraic correctness before using the geometry in the sampler.
    """

    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)

    base_latent = 0.3 * torch.randn(
        (2, 6, 3, 4),
        generator=generator,
        dtype=torch.float64,
    )

    base_blocks = latent_to_local_blocks(base_latent)

    mapped = poincare_expmap0(
        0.2 * base_blocks,
        curvature=1.0,
    )
    recovered = poincare_logmap0(
        mapped,
        curvature=1.0,
    ) / 0.2

    round_trip_error = float(
        torch.max(torch.abs(recovered - base_blocks))
    )

    assert round_trip_error < 1e-10

    zero_modifier = make_hyperbolic_scalar_drift(
        delta=0.0,
        curvature=1.0,
        coordinate_scale=0.2,
    )

    zero_output = zero_modifier(
        x_t=base_latent,
        base_x_prev=base_latent,
        predicted_epsilon=torch.zeros_like(base_latent),
        predicted_x0=torch.zeros_like(base_latent),
        timestep=torch.tensor(0),
        reverse_step=0,
    )

    assert torch.equal(zero_output, base_latent)

    delta = 0.003

    euclidean_modifier = make_euclidean_scalar_drift(delta)

    near_flat_modifier = make_hyperbolic_scalar_drift(
        delta=delta,
        curvature=1e-6,
        coordinate_scale=0.2,
    )

    euclidean_output = euclidean_modifier(
        x_t=base_latent,
        base_x_prev=base_latent,
        predicted_epsilon=torch.zeros_like(base_latent),
        predicted_x0=torch.zeros_like(base_latent),
        timestep=torch.tensor(0),
        reverse_step=0,
    )

    near_flat_output = near_flat_modifier(
        x_t=base_latent,
        base_x_prev=base_latent,
        predicted_epsilon=torch.zeros_like(base_latent),
        predicted_x0=torch.zeros_like(base_latent),
        timestep=torch.tensor(0),
        reverse_step=0,
    )

    flat_limit_error = float(
        torch.max(torch.abs(near_flat_output - euclidean_output))
    )

    assert flat_limit_error < 2e-6

    maximum_ball_norm = float(
        torch.linalg.vector_norm(mapped, dim=-1).max()
    )

    assert maximum_ball_norm < 1.0

    print("Poincare exp/log round-trip: passed")
    print("Zero-drift identity: passed")
    print("Small-curvature Euclidean limit: passed")
    print("Maximum Poincare-ball norm:", maximum_ball_norm)
