import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import numpy as np
import pandas as pd
import torch.nn as nn
import argparse, time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as MAE
import data.build_dataset as build_dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score

train_max, train_min = 0, 0


class NTPP(nn.Module):
    def __init__(self, args, output_layer_size=1):
        super(NTPP, self).__init__()
        self.hidden_size = args['h']  # 隐藏层
        self.num_layers = args['nl']  # Number of RNN Steps
        self.num_sensors = args['num_sensors']
        self.lstm = nn.LSTM(
            input_size=args['element_size'],
            hidden_size=args['h'],
            num_layers=args['nl'],
            batch_first=True
        )
        self.fc = nn.Linear(
            args['h'],
            output_layer_size
        )

    def forward(self, x):
        x = x.view((len(x), 1, -1))
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        _, (lstm_out, _) = self.lstm(x, (h0, c0))
        out = self.fc(lstm_out[0])
        return out


class ShallowRegressionLSTM(nn.Module):

    def __init__(self, embedding_size, num_sensors, hidden_units):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_sensors = num_sensors
        self.hidden_units = hidden_units
        self.num_layers = 1

        self.embedding = nn.Embedding(1000, self.embedding_size)

        self.lstm = nn.LSTM(
            input_size=8,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )
        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()

        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0])  # First dim of Hn is num_layers, which is set to 1 above.
        return out


class SequenceDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x).float()
        self.y = torch.tensor(y).float()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index) -> T_co:
        print(index)
        return self.x[index], self.y[index]


class MultiUserSequenceDataset(Dataset):

    def __init__(self, x, y):
        self.x = torch.tensor(x).float()
        self.y = torch.tensor(y).float()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index) -> T_co:
        return self.x[index], self.y[index]


def parse_args():
    # Arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=10, help='Size of the batch')
    parser.add_argument('--element_size', type=int, default=80, help='Element Size')
    parser.add_argument('--h', type=int, default=128, help='Hidden layer Size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for the optimizer')
    parser.add_argument('--is_cuda', default=False, choices=[True, False], help='CUDA or not')
    parser.add_argument('--optim', default='Adam', choices=['RMS', 'Adam', 'SGD'], help='Optimiser')
    parser.add_argument('--num_sensors', type=int, default=8, help='number of features')
    args = parser.parse_args()
    return args


def train_model(dataloader, model, loss_fun, optimizer):
    num_batches = len(dataloader)
    total_loss = 0
    model.train()
    for x_train, y_train in dataloader:
        optimizer.zero_grad()
        output = model(x_train)
        loss = loss_fun(output, np.squeeze(y_train, axis=1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print("Train average loss:{}".format(avg_loss))


def test_model(input_test, model, loss_fun):
    num_batches = len(input_test)
    total_loss = 0
    model.eval()

    for x_test, y_test in input_test:
        output = model(x_test)
        loss = loss_fun(output, np.squeeze(y_test, axis=1))
        total_loss += loss.item()
    avg_loss = total_loss / num_batches
    print("Test average loss:{}".format(avg_loss))


def predict(data_loader, model):
    output = []
    model.eval()
    with torch.no_grad():
        for X, _ in data_loader:
            y_star = model(X)
            y = y_star.cpu().numpy()
            output.extend(y.astype(float).tolist())

    return output


def eval_model(args, model, test_dataset, label, epoch):
    test_eval_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False)
    y = predict(test_eval_loader, model)
    print("----------------验证集性能评估--------------------")
    print("期望方差 {:.2f}".format(explained_variance_score(label, y)))
    print("MAE {:.2f}".format(mean_absolute_error(label, y)))
    print("MSE {:.2f}".format(mean_squared_error(label, y)))
    print("r2_score {:.2f}".format(r2_score(label, y)))
    if epoch == 16:
        plot(label, y)


def plot(y, pred):
    plt.figure()
    plt.plot(y)
    plt.plot(pred)
    plt.show()


def train_bandwidth():
    args = parse_args()
    args = vars(args)
    model = ShallowRegressionLSTM(6, num_sensors=args['num_sensors'], hidden_units=args['h'])
    x_train, y_train, x_test, y_test = build_dataset.build_bandwidth_dataset()

    train_dataset = MultiUserSequenceDataset(x_train, y_train)
    test_dataset = MultiUserSequenceDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False)

    if args['optim'] == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args['learning_rate'])
    elif args['optim'] == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args['learning_rate'])
    elif args['optim'] == 'RMS':
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=args['learning_rate'])
    else:
        raise Exception('use Proper optimizer')

    loss_function = torch.nn.L1Loss()

    for i in range(args['epochs']):
        print(f"-------------epoch{i}-------------")
        train_model(train_loader, model, loss_function, optimizer)
        test_model(test_loader, model, loss_function)
        if i % 4 == 0:
            eval_model(args, model, test_dataset, y_test, i)


if __name__ == "__main__":
    train_bandwidth()
