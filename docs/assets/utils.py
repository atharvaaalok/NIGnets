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