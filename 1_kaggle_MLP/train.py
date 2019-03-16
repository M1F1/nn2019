from torch.optim import SGD
from torch.nn.functional import cross_entropy
import torch
from models import BaseLine

# some hyperparams
batch_size: int = 64
epoch: int = 3
lr: float = 0.01
momentum: float = 0.9

# prepare data loaders, base don the already loaded datasets
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

# initialize the model
model: BaseLine= BaseLine(input_size=784,
                          hidden_size_1=500,
                          hidden_size_2=100,
                          output_size=10
                          )

# initialize the optimizer
optimizer: torch.optim.Optimizer = SGD(params=model.parameters(),
                                       lr=lr,
                                       momentum=momentum)

# training loop
for e in range(epoch):
    for i, (x, y) in enumerate(train_loader):

        # reset the gradients from previouis iteration
        optimizer.zero_grad()
        # pass through the network
        output: torch.Tensor = model(x)
        # calculate loss
        loss: torch.Tensor = torch.nn.CrossEntropyLoss()(output, y)
        # backward pass thorught the network
        loss.backward()
        # apply the gradients
        optimizer.step()
        # log the loss value
        if (i + 1) % 100 == 0:
            print(f"Epoch {e} iter {i+1}/{len(train_data) // batch_size} loss: {loss.item()}", end="\r")

    # at the end of an epoch run evaluation on the test set
    with torch.no_grad():
        # initialize the number of correct predictions
        correct: int = 0
        for i, (x, y) in enumerate(test_loader):
            # pass through the network
            output: torch.Tensor = model(x)
            # update the number of correctly predicted examples
            pred = output.max(1)[1]
            correct += int(torch.sum(pred == y))

        print(f"\nTest accuracy: {correct / len(test_data)}")