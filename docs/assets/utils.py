import torch
import matplotlib.pyplot as plt


def automate_training(
        model,
        loss_fn,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        epochs: int = 1000,
        print_cost_every: int = 200,
        learning_rate: float = 0.001,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.99)

    for epoch in range(epochs):
        Y_model = model(X_train)
        loss = loss_fn(Y_model, Y_train)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step(loss.item())

        if epoch == 0 or (epoch + 1) % print_cost_every == 0:
            num_digits = len(str(epochs))
            print(f'Epoch: [{epoch + 1:{num_digits}}/{epochs}]. Loss: {loss.item():11.6f}')


def plot_curves(Xc: torch.Tensor, Xt: torch.Tensor) -> None:
    # Get torch tensor to cpu and disable gradient tracking to plot using matplotlib
    Xc = Xc.detach().cpu()
    Xt = Xt.detach().cpu()
    
    plt.fill(Xt[:, 0], Xt[:, 1], color = "#C9C9F5", alpha = 0.46, label = "Target Curve")
    plt.fill(Xc[:, 0], Xc[:, 1], color = "#F69E5E", alpha = 0.36, label = "Candidate Curve")

    plt.plot(Xt[:, 0], Xt[:, 1], color = "#000000", linewidth = 2)
    plt.plot(Xc[:, 0], Xc[:, 1], color = "#000000", linewidth = 2, linestyle = "--")

    plt.axis('equal')
    plt.show()


def plot_surfaces(Xc: torch.Tensor, Xt: torch.Tensor) -> None:
    # Get torch tensor to cpu and disable gradient tracking to plot using matplotlib
    Xc = Xc.detach().cpu()
    Xt = Xt.detach().cpu()

    # Create a figure with a 1x2 grid
    fig, axes = plt.subplots(1, 2, figsize = (12, 6), subplot_kw = {'projection': '3d'})

    # Plot Xt in the first subplot
    axes[0].scatter(Xt[:, 0], Xt[:, 1], Xt[:, 2], s = 5, color = '#1F77B4')
    axes[0].set_title('Target Surface')
    max_dim_t = torch.max(torch.abs(Xt))
    axes[0].set_xlim(-max_dim_t, max_dim_t)
    axes[0].set_ylim(-max_dim_t, max_dim_t)
    axes[0].set_zlim(-max_dim_t, max_dim_t)
    axes[0].set_box_aspect([1, 1, 1])
    axes[0].grid(False)

    # Plot Xc in the second subplot
    axes[1].scatter(Xc[:, 0], Xc[:, 1], Xc[:, 2], s = 5, color = 'r')
    axes[1].set_title('Candidate Surface')
    max_dim_c = torch.max(torch.abs(Xc))
    axes[1].set_xlim(-max_dim_c, max_dim_c)
    axes[1].set_ylim(-max_dim_c, max_dim_c)
    axes[1].set_zlim(-max_dim_c, max_dim_c)
    axes[1].set_box_aspect([1, 1, 1])
    axes[1].grid(False)

    plt.tight_layout()
    plt.show()