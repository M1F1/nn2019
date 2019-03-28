from torch import nn


class BaseLine(nn.Module):
    """
    Simple 2-hidden layer non-linear neural network
    best : p = 0.5
from torch.optim import SGD
from torch.nn.functional import cross_entropy
import torch
from models import BaseLine
import utils_data
import time, datetime
from torch.optim.lr_scheduler import StepLR

torch.manual_seed(42)
# some hyperparams
batch_size: int = 32
epoch: int = 30
lr: float = 0.01
momentum: float = 0.9

# prepare data loaders, base don the already loaded datasets
train_dataset = utils_data.Project1Dataset(data_dir='data',
                                           which='split_train')
val_dataset = utils_data.Project1Dataset(data_dir='data',
                                         which='split_val')

all_data_dataset = utils_data.Project1Dataset(data_dir='data',
                                              which='train')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
all_data_loader = torch.utils.data.DataLoader(all_data_dataset, batch_size=batch_size)

# initialize the model
model: BaseLine = BaseLine(input_size=342,
                           hidden_size_1=2048, # 700
                           hidden_size_2=1024, # 500
                           hidden_size_3=256, # 100
                           output_size=10
                          )

# initialize the optimizer
optimizer: torch.optim.Optimizer = SGD(params=model.parameters(),
                                       lr=lr,
                                       momentum=momentum)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=8, verbose=True)

start = time.time()
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
            print('\r', end='')
            print(f"Epoch {e} iter {i+1}/{len(train_dataset) // batch_size} loss: {loss.item()}", flush=True)

    # at the end of an epoch run evaluation on the test set
    with torch.no_grad():
        model.eval()
        # initialize the number of correct predictions
        correct: int = 0
        for i, (x, y) in enumerate(val_loader):
            # pass through the network
            output: torch.Tensor = model(x)
            # update the number of correctly predicted examples
            pred = output.max(1)[1]
            correct += int(torch.sum(pred == y))

        print(f"\nTest accuracy: {correct / len(val_dataset)}")
    scheduler.step(correct / len(val_dataset))
# TODO: train on all data
duration = time.time() - start
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S.pt')
print(F'training_time last: {duration}')

torch.save(model, F'models/{st}')
    """

    def __init__(self, input_size,
                 hidden_size_1,
                 hidden_size_2,
                 hidden_size_3,
                 output_size):
        super(BaseLine, self).__init__()
        self.linear_layer_1 = nn.Linear(input_size, hidden_size_1)
        self.batch_norm_1 = nn.BatchNorm1d(hidden_size_1)
        self.activation_1 = nn.SELU()
        self.linear_layer_2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.batch_norm_2 = nn.BatchNorm1d(hidden_size_2)
        self.activation_2 = nn.SELU()
        self.linear_layer_3 = nn.Linear(hidden_size_2, hidden_size_3)
        self.batch_norm_3 = nn.BatchNorm1d(hidden_size_3)
        self.activation_3 = nn.SELU()
        self.linear_layer_4 = nn.Linear(hidden_size_3, output_size)
        self.batch_norm_4 = nn.BatchNorm1d(output_size)
        self.dropout_1 = nn.Dropout(p=0.3) # 0.3
        self.dropout_2 = nn.Dropout(p=0.5) # 0.3
        self.dropout_3 = nn.Dropout(p=0.7) # 0.3

    def forward(self, x):
        d_1 = self.dropout_1(
                self.activation_1(
                    self.batch_norm_1(
                        self.linear_layer_1(x))))
        d_2 = self.dropout_2(
            self.activation_2(
                self.batch_norm_2(
                    self.linear_layer_2(d_1))))
        d_3 = self.dropout_3(
                self.activation_3(
                    self.batch_norm_3(
                        self.linear_layer_3(d_2))))
        l_4 = self.batch_norm_4(self.linear_layer_4(d_3))
        return l_4
