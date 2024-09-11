import torch
import torch.nn as nn
from torch.autograd import Variable
from tabulate import tabulate
import sys
import os
# solve the problem of no module named 'models' in the following line
file_path = os.path.abspath(__file__)
file_path = file_path[:file_path.find(r'/models')]
sys.path.insert(0, file_path)
from models.data_prepare import *

class MLP(nn.Module):
    def __init__(self, input_size, output_size, gru_hidden_size, mlp_hidden_size, mlp_layer, device=0, output_last=True, x_flag=False, use_gpu=True):

        super(MLP, self).__init__()

        self.input_size = input_size
        self.gru_hidden_size = gru_hidden_size
        self.delta_size = input_size
        self.mask_size = input_size
        self.output_size = output_size
        self.device = device
        self.use_gpu = use_gpu
        self.mlp_hidden_size = mlp_hidden_size

        if self.use_gpu and self.device == 0:
            self.zeros = Variable(torch.zeros(input_size).cuda())
        if self.use_gpu and self.device == 1:
            cuda1 = torch.device('cuda:1')
            self.zeros = Variable(torch.zeros(input_size).cuda(device=cuda1))
        if not self.use_gpu:
            self.zeros = Variable(torch.zeros(input_size))

        self.linear = nn.Sequential(
            nn.Linear(self.input_size * 2, self.gru_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.gru_hidden_size, self.gru_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.gru_hidden_size, self.gru_hidden_size),
            nn.LeakyReLU()
        )
        self.fc = nn.Linear(gru_hidden_size, output_size)

    def step(self, x, h, mask):
        h = self.gru_cell(torch.cat([x, mask], dim=1), h)
        return h

    def forward(self, input):
        batch_size = input.size(0)
        step_size = input.size(2)

        x = torch.squeeze(input[:, 0, :, :])  # batch_size, time_steps, var_size
        x[x != x] = -1  # replace nan into -1
        mask = torch.squeeze(input[:, 2, :, :])

        output = self.linear(torch.cat([x, mask], dim=1))
        return self.fc(output), None

if __name__ == "__main__":
    # Get the current file path
    current_file_path = os.path.abspath(__file__)
    # Change the working directory to the current file's directory
    os.chdir(os.path.dirname(current_file_path))
    root_path = "../../StepCountsDataset/"
    granularity = 3
    final_train_labels, test_labels, train_x, test_x, raw_train_x, raw_test_x = prepare_mixed_data_granularity(root_path, granularity)
    beta = 0.9
    normalized_weights = balance_loss_weight(final_train_labels, 70, beta=beta)
    # Create a table with the print statements
    table = [
        ['len(final_train_labels)', len(final_train_labels)],
        ['len(test_labels)', len(test_labels)],
        ['train_x.shape', train_x.shape],
        ['test_x.shape', test_x.shape]
    ]
    # Print the table
    print(tabulate(table, headers=['Variable', 'Value'], tablefmt='grid'))

    train_loader, test_loader = prepare_train_test_data(train_data=train_x,
                                                        test_data=test_x,
                                                        train_labels=final_train_labels,
                                                        test_labels=test_labels,
                                                        raw_train_data=raw_train_x,
                                                        raw_test_data=raw_test_x,
                                                        batch_size=100)
    # mlp_hidden_size is 256, which is determined by the validation set in original paper. Since we only provide about one-fifth of data in paper, this hyperparameter should be reconsidered. 
    mlp_gru = MLP(input_size=int(576/2), gru_hidden_size=64, mlp_hidden_size=576*2, mlp_layer=1, output_size=70, use_gpu=True,
                      device=0, output_last=True)
    train_model(mlp_gru, train_loader, test_loader, text_path='result_MLP.txt', model_path='result_MLP.pt',
                device=0, num_epochs=100, learning_rate_decay=10, use_gpu=True, classes=70,
                loss_weights=normalized_weights)
