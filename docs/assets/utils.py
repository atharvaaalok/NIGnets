import torch


def automate_training(
        model,
        loss_fn,
        X_train,
        Y_train,
        epochs = 1000,
        print_cost_every = 200,
        learning_rate = 0.001,
):
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