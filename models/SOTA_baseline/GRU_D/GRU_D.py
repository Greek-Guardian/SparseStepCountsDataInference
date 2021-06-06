import torch
import torch.nn as nn
from torch.autograd import Variable
from models.data_prepare import *


class GRU_D(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, device=0, output_last=True, x_flag=False, use_gpu=True):

        super(GRU_D, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
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

        self.gamma_x_l = nn.Linear(self.delta_size, self.delta_size)
        self.gamma_h_l = nn.Linear(self.delta_size, self.hidden_size)
        self.gru_cell = nn.GRUCell(self.input_size * 2, self.hidden_size)

        self.fc = nn.Linear(hidden_size, output_size)

    def step(self, x, x_last_ob, x_next_ob, h, mask, d_forward):

        delta_x = torch.exp(-torch.max(self.zeros, self.gamma_x_l(d_forward)))
        delta_h = torch.exp(-torch.max(self.zeros, self.gamma_h_l(d_forward)))

        x = mask * x + (1 - mask) * (delta_x * x_last_ob + (1 - delta_x) * x_next_ob)
        inputs = torch.cat([x, mask], dim=1)
        h = delta_h * h

        h = self.gru_cell(inputs, h)

        # combined = torch.cat((x, h, mask), 1)
        # z = torch.sigmoid(self.zl(combined))
        # r = torch.sigmoid(self.rl(combined))
        # combined_r = torch.cat((x, r * h, mask), 1)
        # h_tilde = torch.tanh(self.hl(combined_r))
        # h = (1 - z) * h + z * h_tilde

        return h, x

    def forward(self, input):
        # batch number
        batch_size = input.size(0)
        # 6 types: x, x_last_ob, mask, d_forward, x_next_ob, d_backward
        type_size = input.size(1)
        # time steps: every 5 mins 288
        step_size = input.size(2)

        hidden_state = self.init_hidden(batch_size)
        xs = torch.squeeze(input[:, 0, :, :])  # batch_size, time_steps, var_size
        xs[xs != xs] = -1  # replace nan into -1
        x_last_obs = torch.squeeze(input[:, 1, :, :])
        masks = torch.squeeze(input[:, 2, :, :])
        d_forwards = torch.squeeze(input[:, 3, :, :])
        x_next_obs = torch.squeeze(input[:, 4, :, :])

        outputs = None
        xs_imputation = None
        for i in range(step_size):
            hidden_state, x_im = self.step(xs[:, i:i + 1],
                                           x_last_obs[:, i:i + 1],
                                           x_next_obs[:, i:i + 1],
                                           hidden_state,
                                           masks[:, i:i + 1],
                                           d_forwards[:, i:i + 1])
            if xs_imputation is None:
                xs_imputation = x_im
            else:
                xs_imputation = torch.cat((xs_imputation, x_im), 1)

        if self.output_last and not self.x_flag:
            return self.fc(hidden_state)
            # return self.fc(outputs[:, -1, :])  # batch_size, hidden_size
        if self.output_last and self.x_flag:
            return self.fc(hidden_state), xs_imputation
            # return self.fc(outputs[:, -1, :]), xs_imputation
        else:
            return outputs  # batch_size, time_steps, hidden_size

    def init_hidden(self, batch_size):
        use_gpu = self.use_gpu
        if use_gpu and self.device == 0:
            hidden_state = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            return hidden_state
        elif use_gpu and self.device == 1:
            cuda1 = torch.device('cuda:1')
            hidden_state = Variable(torch.zeros(batch_size, self.hidden_size).cuda(device=cuda1))
            return hidden_state
        else:
            hidden_state = Variable(torch.zeros(batch_size, self.hidden_size))
            return hidden_state


if __name__ == "__main__":
    root_path = "../../../data/"
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

    grud = GRU_D(input_size=1, output_size=70, hidden_size=64, device=0, use_gpu=True)
    train_model(grud, train_loader, test_loader, text_path='result.txt', model_path='result.pt',
                device=0, num_epochs=100, learning_rate_decay=10, use_gpu=True, classes=70,
                loss_weights=normalized_weights)