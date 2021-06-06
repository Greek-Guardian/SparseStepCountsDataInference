import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from models.data_prepare import *


class TBM_GRU(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, tau, alpha, device=0, output_last=True, x_flag=False, use_gpu=True):

        super(TBM_GRU, self).__init__()

        self.hidden_size = hidden_size
        self.delta_size = input_size
        self.mask_size = input_size
        self.output_size = output_size
        self.device = device
        self.x_flag = x_flag
        self.output_last = output_last
        self.use_gpu = use_gpu
        self.tau = tau
        self.alpha = alpha

        if self.use_gpu and self.device == 0:
            self.identity = torch.eye(input_size).cuda()
            self.old_weight = torch.ones(3*hidden_size, 2).cuda()

        if self.use_gpu and self.device == 1:
            cuda1 = torch.device('cuda:1')
            self.identity = torch.eye(input_size).cuda(device=cuda1)
            self.old_weight = torch.ones(3*hidden_size, 2).cuda(device=cuda1)

        if not self.use_gpu:
            self.identity = torch.eye(input_size)
            self.old_weight = torch.ones(3*hidden_size, 2)

        self.relu = nn.ReLU()
        self.tbm = nn.Linear(input_size, input_size)
        # input + mask = input of GRU
        self.gru = nn.GRU(input_size=input_size+1, hidden_size=hidden_size, batch_first=True)
        # initialize old weights are ones
        # self.old_weight = 0
        for para in list(self.gru.named_parameters()):
            if para[0] == "weight_ih_l0":
                self.current_weight = para[1]
                break
        # self.current_weight = self.gru.weight_ih_l
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        # batch number
        batch_size = input.size(0)
        # 6 types: x, x_last_ob, mask, d_forward, x_next_ob, d_backward
        type_size = input.size(1)
        # time steps: every 5 mins 288
        step_size = input.size(2)
        # variable dimension : 1
        spatial_size = input.size(3)

        # shape of h0 (num_layers * num_directions, batch, hidden_size)
        hidden_state = self.init_hidden(batch_size)
        # the shape of x and masks are batch_size * 288
        xs = torch.squeeze(input[:, 0, :, :])
        xs[xs != xs] = -1  # replace nan into -1
        masks = torch.squeeze(input[:, 2, :, :])
        old_tau = self.tau

        grads = (self.current_weight - self.old_weight).mean()
        self.tau += grads * self.alpha
        if self.tau >= 1 and self.tau != old_tau:
            xs = self.belief_filling(self.tau, xs, masks)

        self.old_weight = self.current_weight
        for para in list(self.gru.named_parameters()):
            if para[0] == "weight_ih_l0":
                self.current_weight = para[1]
                break

        # self.tau = self.relu(self.tbm(old_tau))

        # put xs and masks into one tensor with shape(batch_size, step_size, 2)
        gru_input = torch.stack((xs, masks), 2)
        gru_output, hn = self.gru(gru_input, hidden_state)
        output = self.fc(torch.squeeze(gru_output[:, -1, :]))

        return output

    def belief_filling(self, tau, x, mask):
        half_tau = torch.round(tau/2)
        batch_size = x.size(0)
        step_size = x.size(1)
        for i in range(batch_size):
            oneday_x = x[i, :]
            oneday_m = mask[i, :]
            # effective data point indexs
            e_ind = (oneday_m==1).nonzero(as_tuple=True)[0]
            for j in range(e_ind.size(0)):
                if j == 0:
                    if e_ind[j] <= half_tau:
                        window_left = 0
                    else:
                        window_left = e_ind[j] - half_tau

                if j > 0:
                    if e_ind[j] - e_ind[j-1] > half_tau:
                        window_left = e_ind[j] - half_tau
                    else:
                        window_left = e_ind[j-1] + 1

                if j == e_ind.size(0):
                    if step_size - e_ind[j] - 1 <= half_tau:
                        window_right = step_size - 1
                    else:
                        window_right = e_ind[j] + half_tau

                if j < e_ind.size(0):
                    if e_ind[j+1] - e_ind[j] > half_tau:
                        window_right = e_ind[j] + half_tau
                    else:
                        window_right = e_ind[j+1] - 1

                oneday_x[window_left: window_right+1] = oneday_x[e_ind[j]]
            x[i, :] = oneday_x

        return x

    def init_hidden(self, batch_size):
        use_gpu = self.use_gpu
        if use_gpu and self.device == 0:
            hidden_state = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
            return hidden_state
        elif use_gpu and self.device == 1:
            cuda1 = torch.device('cuda:1')
            hidden_state = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda(device=cuda1))
            return hidden_state
        else:
            hidden_state = Variable(torch.zeros(1, batch_size, self.hidden_size))
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

    tbm = TBM_GRU(input_size=1, output_size=70, hidden_size=64, tau=10, alpha=1000, device=0, use_gpu=True)
    train_model(tbm, train_loader, test_loader, text_path='result.txt', model_path='result.pt',
                device=0, num_epochs=100, learning_rate_decay=10, use_gpu=True, classes=70,
                loss_weights=normalized_weights)






