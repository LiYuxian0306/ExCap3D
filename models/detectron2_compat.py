"""
Compatibility layer for detectron2 functions.
Replaces detectron2 dependencies to avoid version conflicts.
"""
import torch
import torch.nn.functional as F
import torch.distributed as dist
from typing import Callable, Tuple


def get_world_size() -> int:
    """
    Get the number of processes in the distributed group.
    Returns 1 if not in distributed mode.
    """
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def point_sample(
    input: torch.Tensor,
    point_coords: torch.Tensor,
    align_corners: bool = False,
) -> torch.Tensor:
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 2D point
    sampling on 4D (or higher) tensors.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features
            map on a H x W spatial grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains coordinates 
            of points at which the input should be sampled. Coordinates are in [0, 1].
        align_corners (bool): If align_corners=True, the extrema (-1 and 1) are
            considered as referring to the center points of the input's corner pixels.
            If False, they are instead considered as referring to the corner points of
            the input's corner pixels, making the sampling more resolution agnostic.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) that contains sampled features 
            for the points with the same data type as `input`.
    """
    # input: (N, C, H, W)
    # point_coords: (N, P, 2) in [0, 1] range
    
    # Convert point_coords from [0, 1] to [-1, 1] for grid_sample
    point_coords = point_coords * 2.0 - 1.0
    
    # Reshape for grid_sample: (N, P, 2) -> (N, 1, P, 2)
    # grid_sample expects (N, H_out, W_out, 2)
    point_coords = point_coords.unsqueeze(1)  # (N, 1, P, 2)
    
    # grid_sample: input (N, C, H, W), grid (N, 1, P, 2) -> output (N, C, 1, P)
    output = F.grid_sample(
        input,
        point_coords,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=align_corners,
    )
    
    # Remove the spatial dimension: (N, C, 1, P) -> (N, C, P)
    output = output.squeeze(2)
    
    return output


def get_uncertain_point_coords_with_randomness(
    coarse_logits: torch.Tensor,
    uncertainty_func: Callable,
    num_points: int,
    oversample_ratio: int,
    importance_sample_ratio: float,
) -> torch.Tensor:
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty.
    The unceratinties are calculated for each point using 'uncertainty_func'.

    See PointRend paper for details.

    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, H, W) or (N, 1, H, W) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P)
            that contains logit predictions at P points and returns their
            uncertainties as a Tensor of shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via
            importance sampling.

    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the
            coordinates of P sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points with
    # high uncertainties can be a major performance bottleneck.
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )
    if num_random_points > 0:
        point_coords = torch.cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device),
            ],
            dim=1,
        )
    return point_coords

