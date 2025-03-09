import torch
from .shape_svg.svg_extract_xy import svg_extract_xy
import open3d as o3d
import numpy as np


def circle(num_pts: int) -> torch.Tensor:
    theta = torch.linspace(0, 2 * torch.pi, num_pts)

    # Compute x and y coordinates and concatenate to form a matrix
    X = torch.stack([torch.cos(theta), torch.sin(theta)], dim = 1)
    return X


def square(num_pts: int) -> torch.Tensor:
    # Generate points on the unit circle first and then map them to a square
    theta = torch.linspace(0, 2 * torch.pi, num_pts)

    x, y = torch.cos(theta), torch.sin(theta)
    s = torch.maximum(torch.abs(x), torch.abs(y))
    x_sq, y_sq = x/s, y/s

    X = torch.stack([x_sq, y_sq], dim = 1)
    return X


def square_from_t(t: torch.Tensor) -> torch.Tensor:
    # Generate theta values corresponding to t
    theta = 2 * torch.pi * t.reshape(-1)
    
    x, y = torch.cos(theta), torch.sin(theta)
    s = torch.maximum(torch.abs(x), torch.abs(y))
    x_sq, y_sq = x/s, y/s

    X = torch.stack([x_sq, y_sq], dim = 1)
    return X


def stanford_bunny(num_pts: int) -> torch.Tensor:
    X = svg_extract_xy('stanford_bunny.svg', num_pts = num_pts)
    return X


def airfoil(num_pts: int) -> torch.Tensor:
    X = svg_extract_xy('airfoil.svg', num_pts = num_pts)
    return X


def heart(num_pts: int) -> torch.Tensor:
    X = svg_extract_xy('heart.svg', num_pts = num_pts)
    return X


def hand(num_pts: int) -> torch.Tensor:
    X = svg_extract_xy('hand.svg', num_pts = num_pts)
    return X


def puzzle_piece(num_pts: int) -> torch.Tensor:
    X = svg_extract_xy('puzzle_piece.svg', num_pts = num_pts)
    return X


def airplane(num_pts: int) -> torch.Tensor:
    X = svg_extract_xy('airplane.svg', num_pts = num_pts)
    return X


def snowflake_fractal(num_pts: int) -> torch.Tensor:
    X = svg_extract_xy('snowflake_fractal.svg', num_pts = num_pts)
    return X


def star_fractal(num_pts: int) -> torch.Tensor:
    X = svg_extract_xy('star_fractal.svg', num_pts = num_pts)
    return X


def minkowski_fractal(num_pts: int) -> torch.Tensor:
    X = svg_extract_xy('minkowski_fractal.svg', num_pts = num_pts)
    return X


def sphere(num_pts: int) -> torch.Tensor:
    """Generates approximately evenly distributed points on the unit sphere using the Fibonacci
    lattice.
    """
    idx = torch.arange(0, num_pts) + 0.5
    phi = torch.arccos(1 - 2 * idx / num_pts)
    theta = torch.pi * (1 + 5**0.5) * idx

    # Compute x, y and z coordinates
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)

    # Concatenate to form a matrix of shape: (num_pts, 3)
    X = torch.stack([x, y, z], dim = 1)
    return X


def cube(num_pts: int) -> torch.Tensor:
    # Generate points on the unit sphere first and then map them to a cube
    idx = torch.arange(0, num_pts) + 0.5
    phi = torch.arccos(1 - 2 * idx / num_pts)
    theta = torch.pi * (1 + 5**0.5) * idx

    # Compute x, y and z coordinates
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)

    # Compute the max-norm projection to map the sphere to the cube
    sphere_pts = torch.stack([x, y, z], dim = 1)
    s, _ = torch.max(torch.abs(sphere_pts), dim = 1)
    x_cube, y_cube, z_cube = x/s, y/s, z/s

    X = torch.stack([x_cube, y_cube, z_cube], dim = 1)
    return X


def torus(num_pts: int) -> torch.Tensor:
    # Create angular grid
    uv_grid = torch.rand(num_pts, 2) * (2 * torch.pi)
    u, v = uv_grid[:, 0], uv_grid[:, 1]

    # Parameteric equation of torus
    R, r = 3, 1
    x = (R + r * torch.cos(v)) * torch.cos(u)
    y = (R + r * torch.cos(v)) * torch.sin(u)
    z = r * torch.sin(v)

    X = torch.stack([x, y, z], dim = 1)
    return X


def stanford_bunny_3d(num_pts: int) -> torch.Tensor:
    # Get the bunny mesh
    bunny = o3d.data.BunnyMesh()
    mesh = o3d.io.read_triangle_mesh(bunny.path)
    mesh.compute_vertex_normals()

    # Get point cloud
    pcd = mesh.sample_points_poisson_disk(number_of_points = num_pts, init_factor = 5)

    points = np.asarray(pcd.points)
    X = torch.tensor(points, dtype = torch.float32)

    # Center at origin and normalize
    X = X - X.mean(dim = 0)
    X = X / torch.max(torch.abs(X))

    # Define rotation angles as tensors
    theta_z = torch.tensor(torch.pi / 2)
    theta_x = torch.tensor(-torch.pi * 7/4)
    theta_y = torch.tensor(torch.pi / 2)

    # Rotation matrices
    R_z = torch.tensor([
        [torch.cos(theta_z), -torch.sin(theta_z), 0],
        [torch.sin(theta_z),  torch.cos(theta_z), 0],
        [0, 0, 1]
    ])

    R_x = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(theta_x), -torch.sin(theta_x)],
        [0, torch.sin(theta_x),  torch.cos(theta_x)]
    ])

    R_y = torch.tensor([
        [torch.cos(theta_y), 0, torch.sin(theta_y)],
        [0, 1, 0],
        [-torch.sin(theta_y), 0, torch.cos(theta_y)]
    ])

    # Combined rotation: first z, then x, then y
    R = R_y @ R_x @ R_z
    X = (R @ X.T).T

    return X