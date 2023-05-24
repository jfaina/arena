import os
import sys
import torch as t
from torch import Tensor
import einops
from ipywidgets import interact
import plotly.express as px
from ipywidgets import interact
from pathlib import Path
from IPython.display import display
from jaxtyping import Float, Int, Bool, Shaped, jaxtyped
import typeguard

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part1_ray_tracing"
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

from plotly_utils import imshow
from part1_ray_tracing.utils import render_lines_with_plotly, setup_widget_fig_ray, setup_widget_fig_triangle
import part1_ray_tracing.tests as tests

MAIN = __name__ == "__main__"

def make_rays_1d(num_pixels: int, y_limit: float) -> t.Tensor:
    '''
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) where the num_points dimension contains (origin, direction) and the num_dim dimension contains xyz.

    Example of make_rays_1d(9, 1.0): [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    '''
    rays = t.zeros((num_pixels, 2, 3), dtype=t.float32)
    rays[:, 1, 0] = 1.0
    rays[:, 1, 1] = t.linspace(-y_limit, y_limit, num_pixels)
    return rays

@jaxtyped
@typeguard.typechecked
def intersect_ray_1d(ray: Float[t.Tensor, "npoints=2 ndim=3"], segment: Float[t.Tensor, "npoints=2 ndim=3"]) -> bool:
    '''
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    '''
    ray_2d = ray[:,:-1]
    segment_2d = segment[:,:-1]
    A = t.stack((ray_2d[1], segment_2d[0] - segment_2d[1]), dim=1)
    b = segment_2d[0] - ray_2d[0]
    try:
        x = t.linalg.solve(A, b)
        u, v = x[0].item(), x[1].item()
        return u >= 0 and v >= 0 and v <= 1
    except Exception:
        return False

@jaxtyped
@typeguard.typechecked
def intersect_rays_1d(rays: Float[Tensor, "nrays 2 3"], segments: Float[Tensor, "nsegments 2 3"]) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, does it intersect any segment.
    '''
    rays_xy = rays[:, :, :-1]
    segments_xy = segments[:, :, :-1]
    nrays = rays.shape[0]
    nsegments = segments.shape[0]
    rays_all = einops.repeat(rays_xy, 'nrays b c -> nrays nsegments b c', nsegments=nsegments)
    segments_all = einops.repeat(segments_xy, 'nsegments b c -> nrays nsegments b c', nrays=nrays)
    D = rays_all[:, :, 1]
    L = segments_all[:, :, 0] - segments_all[:, :, 1]
    similarity = t.abs(t.cosine_similarity(D, L, dim=-1))
    invertible = t.logical_not(t.isclose(similarity, t.ones((nrays, nsegments))))
    A = t.stack((D, L), dim=-1)
    b = segments_all[:, :, 0] - rays_all[:, :, 0]
    x = t.linalg.solve(A[invertible], b[invertible])
    intersects = t.zeros((nrays, nsegments), dtype=t.bool)
    intersects[invertible] = (x[..., 0] >= 0) & (x[..., 1] >= 0) & (x[..., 1] <= 1)
    ret = intersects.any(dim=1)
    return ret

def make_rays_2d(num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float) -> Float[t.Tensor, "nrays 2 3"]:
    '''
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    '''
    pass

Point = Float[Tensor, "points=3"]

@jaxtyped
@typeguard.typechecked
def triangle_ray_intersects(A: Point, B: Point, C: Point, O: Point, D: Point) -> bool:
    '''
    A: shape (3,), one vertex of the triangle
    B: shape (3,), second vertex of the triangle
    C: shape (3,), third vertex of the triangle
    O: shape (3,), origin point
    D: shape (3,), direction point

    Return True if the ray and the triangle intersect.
    '''
    pass