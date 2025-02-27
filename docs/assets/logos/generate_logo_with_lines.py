import torch
import matplotlib.pyplot as plt

from svg_extract_xy import svg_extract_xy


def square(num_pts: int) -> torch.Tensor:
    X = svg_extract_xy('logo_square.svg', num_pts = num_pts)
    n = num_pts // 4 - 1
    X = torch.vstack([X[-n:], X[:-n]])
    return X


def curve(num_pts: int) -> torch.Tensor:
    X = svg_extract_xy('logo_curve.svg', num_pts = num_pts)
    return X


def plot_curves_with_lines(Xc: torch.Tensor, Xt: torch.Tensor) -> None:
    # Move to CPU and detach gradients for plotting
    Xc = Xc.detach().cpu()
    Xt = Xt.detach().cpu()
    
    # Fill the shapes
    plt.fill(Xt[:, 0], Xt[:, 1], color = "#C9C9F5", alpha = 0.46, label = "Target Curve")
    plt.fill(Xc[:, 0], Xc[:, 1], color = "#F69E5E", alpha = 0.36, label = "Candidate Curve")
    
    # Outline the shapes
    plt.plot(Xt[:, 0], Xt[:, 1], color = "#000000", linewidth = 2)
    plt.plot(Xc[:, 0], Xc[:, 1], color = "#000000", linewidth = 2, linestyle = "--")

    # Draw line segments from each point on Xc to the corresponding point on Xt
    for i in range(0, len(Xc), 15):
        plt.plot(
            [Xc[i, 0], Xt[i, 0]],
            [Xc[i, 1], Xt[i, 1]],
            color = "black",
            linewidth = 0.5
        )

    plt.axis("equal")
    plt.savefig("logo_with_lines.svg", format = "svg")
    plt.show()


# Create points on the square and curve
num_pts = 1000
X_square = square(num_pts = num_pts)
X_curve = curve(num_pts = num_pts)

plot_curves_with_lines(X_curve, X_square)