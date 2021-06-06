import torch
import torch.nn as nn
from torch.autograd import Variable
from models.data_prepare import *


class MLP_GRU(nn.Module):
    def __init__(self, input_size, output_size, gru_hidden_size, mlp_hidden_size, mlp_layer, device=0, output_last=True, x_flag=False, use_gpu=True):

        super(MLP_GRU, self).__init__()

        self.input_size = input_size
        self.gru_hidden_size = gru_hidden_size
        self.mlp_hidden_size = mlp_hidden_size
        self.mlp_layer = mlp_layer
        self.delta_size = input_size
        self.mask_size = input_size
        self.output_size = output_size
        self.device = device
        self.x_flag = x_flag
        self.output_last = output_last
        self.use_gpu = use_gpu

        if self.use_gpu and self.device == 0:
            self.zeros = Variable(torch.zeros(input_size).cuda())
        if self.use_gpu and self.device == 1:
            cuda1 = torch.device('cuda:1')
            self.zeros = Variable(torch.zeros(input_size).cuda(device=cuda1))
        if not self.use_gpu:
            self.zeros = Variable(torch.zeros(input_size))

        self.gamma_x_l = nn.Linear(self.mlp_hidden_size, self.delta_size)
        self.gru_cell = nn.GRUCell(self.input_size * 2, self.gru_hidden_size)
        self.mlp_model = nn.Sequential(nn.Linear(self.input_size*3, self.mlp_hidden_size), nn.ReLU())
        if self.mlp_layer > 1:
            for i in range(self.mlp_layer - 1):
                self.mlp_model.add_module("linear{}".format(i), nn.Linear(self.mlp_hidden_size, mlp_hidden_size))
                self.mlp_model.add_module("active{}".format(i), nn.ReLU())

        self.fc = nn.Linear(gru_hidden_size, output_size)

    def step(self, time_step, x, x_last_ob, x_next_ob, h, mask, d_forward, d_backward):

        # mlp_input = torch.cat((time_step, d_forward, d_backward), dim=1)
        # mlp_h = self.mlp_model(mlp_input)
        gamma_x = torch.sigmoid(self.gamma_x_l(self.mlp_model(torch.cat((time_step, d_forward, d_backward), dim=1))))

        x = mask * x + (1 - mask) * (gamma_x * x_last_ob + (1 - gamma_x) * x_next_ob)
        # inputs = torch.cat([x, mask], dim=1)

        # combined = torch.cat((x, h, mask), 1)
        # z = torch.sigmoid(self.zl(combined))
        # r = torch.sigmoid(self.rl(combined))
        # combined_r = torch.cat((x, r * h, mask), 1)
        # h_tilde = torch.tanh(self.hl(combined_r))
        # h = (1 - z) * h + z * h_tilde
        h = self.gru_cell(torch.cat([x, mask], dim=1), h)

        return h, x

    def forward(self, input):
        # batch number
        batch_size = input.size(0)
        # 6 types: x, x_last_ob, mask, d_forward, x_next_ob, d_backward
        # type_size = input.size(1)
        # time steps: every 5 mins 288
        step_size = input.size(2)
        # variable dimension : 1
        # spatial_size = input.size(3)

        hidden_state, time_steps = self.init_hidden(batch_size)
        xs = torch.squeeze(input[:, 0, :, :])  # batch_size, time_steps, var_size
        xs[xs != xs] = -1  # replace nan into -1
        x_last_obs = torch.squeeze(input[:, 1, :, :])
        masks = torch.squeeze(input[:, 2, :, :])
        d_forwards = torch.squeeze(input[:, 3, :, :])
        x_next_obs = torch.squeeze(input[:, 4, :, :])
        d_backwards = torch.squeeze(input[:, 5, :, :])

        outputs = None
        xs_imputation = None
        for i in range(step_size):
            hidden_state, x_im = self.step(time_steps[:, i:i+1],
                                           xs[:, i:i+1],
                                           x_last_obs[:, i:i+1],
                                           x_next_obs[:, i:i+1],
                                           hidden_state,
                                           masks[:, i:i+1],
                                           d_forwards[:, i:i+1],
                                           d_backwards[:, i:i+1])

        #     if xs_imputation is None:
        #         xs_imputation = x_im
        #     else:
        #         xs_imputation = torch.cat((xs_imputation, x_im), 1)
        #
        #     fc_output = self.fc(hidden_state)
        #     if outputs is None:
        #         outputs = fc_output.unsqueeze(1)
        #     else:
        #         outputs = torch.cat((outputs, fc_output.unsqueeze(1)), 1)
        #
        # if self.output_last and not self.x_flag:
        #     return outputs[:, -1, :]
        # if self.output_last and self.x_flag:
        #     return outputs[:, -1, :], xs_imputation
        # else:
        #     return outputs  # batch_size, time_steps, output_size
        return self.fc(hidden_state)

    def init_hidden(self, batch_size):
        time_step_lists = []
        for i in range(batch_size):
            time_step_lists.append(list(range(288)))
        use_gpu = self.use_gpu
        if use_gpu and self.device == 0:
            time_step = torch.Tensor(time_step_lists).cuda()
            hidden_state = Variable(torch.zeros(batch_size, self.gru_hidden_size).cuda())

        elif use_gpu and self.device == 1:
            cuda1 = torch.device('cuda:1')
            time_step = torch.Tensor(time_step_lists).cuda(device=cuda1)
            hidden_state = Variable(torch.zeros(batch_size, self.gru_hidden_size).cuda(device=cuda1))

        else:
            time_step = torch.Tensor(time_step_lists)
            hidden_state = Variable(torch.zeros(batch_size, self.gru_hidden_size))
        return hidden_state, time_step


if __name__ == "__main__":
    root_path = "../../data/"
    granularity = 3
    final_train_labels, test_labels, train_x, test_x = prepare_mixed_data_granularity(root_path, granularity)
    print(len(final_train_labels))
    print(len(test_labels))
    print(train_x.shape)
    print(test_x.shape)
    beta = 0.9
    normalized_weights = balance_loss_weight(final_train_labels, 70, beta=beta)
    print(normalized_weights)
    train_loader, test_loader = prepare_train_test_data(train_data=train_x,
                                                        test_data=test_x,
                                                        train_labels=final_train_labels,
                                                        test_labels=test_labels,
                                                        batch_size=100)
    mlp_gru = MLP_GRU(input_size=1, gru_hidden_size=64, mlp_hidden_size=256, mlp_layer=1, output_size=70, use_gpu=True,
                      device=0, output_last=True)
    train_model(mlp_gru, train_loader, test_loader, text_path='result.txt', model_path='result.pt',
                device=0, num_epochs=100, learning_rate_decay=10, use_gpu=True, classes=70,
                loss_weights=normalized_weights)
