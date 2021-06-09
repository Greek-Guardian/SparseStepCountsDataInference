import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import autograd
from torch.nn.parameter import Parameter
import math
import time
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from models.data_prepare import *


class TemporalDecay(nn.Module):
    # decay factor for hidden state

    def __init__(self, input_size, rnn_hid_size):
        super(TemporalDecay, self).__init__()
        self.rnn_hid_size = rnn_hid_size
        self.build(input_size)

    def build(self, input_size):
        self.W = Parameter(torch.Tensor(self.rnn_hid_size, input_size))
        self.b = Parameter(torch.Tensor(self.rnn_hid_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma


class BRITSI(nn.Module):

    def __init__(self, input_size, rnn_hid_size, output_size, device=0, use_gpu=True):
        super(BRITSI, self).__init__()

        self.input_size = input_size
        self.rnn_hid_size = rnn_hid_size
        self.output_size = output_size
        self.device = device
        self.use_gpu = use_gpu
        # self.x_loss = nn.MSELoss(reduction='sum')
        self.gamma_h_f = nn.Linear(self.input_size, self.rnn_hid_size)
        self.gamma_h_b = nn.Linear(self.input_size, self.rnn_hid_size)

        self.fc_f = nn.Linear(rnn_hid_size, output_size)
        self.fc_b = nn.Linear(rnn_hid_size, output_size)

        # if self.use_gpu and self.device == 0:
        #     self.identity = torch.eye(input_size).cuda()
        #     self.zeros = Variable(torch.zeros(input_size).cuda())
        # if self.use_gpu and self.device == 1:
        #     cuda1 = torch.device('cuda:1')
        #     self.identity = torch.eye(input_size).cuda(device=cuda1)
        #     self.zeros = Variable(torch.zeros(input_size).cuda(device=cuda1))
        # if not self.use_gpu:
        #     self.identity = torch.eye(input_size)
        #     self.zeros = Variable(torch.zeros(input_size))

        self.rnn_cell_f = nn.GRUCell(self.input_size * 2, self.rnn_hid_size)
        self.regression_f = nn.Linear(self.rnn_hid_size, self.input_size)
        self.rnn_cell_b = nn.GRUCell(self.input_size * 2, self.rnn_hid_size)
        self.regression_b = nn.Linear(self.rnn_hid_size, self.input_size)
        # self.temp_decay = TemporalDecay(input_size=self.input_size, rnn_hid_size=self.rnn_hid_size)

    def forward(self, input):
        # prepare data:
        # batch number
        batch_size = input.size(0)
        # time steps: every 5 mins 288
        step_size = input.size(2)
        time_step_list_f = list(range(step_size))
        # delta to last observation
        d_f = torch.squeeze(input[:, 3, :, :])
        # delta to next observation
        d_b = torch.squeeze(input[:, 5, :, :])
        time_step_list_b = list(reversed(time_step_list_f))
        xs = torch.squeeze(input[:, 0, :, :])
        xs[xs != xs] = -1
        masks = torch.squeeze(input[:, 2, :, :])

        # forward imputation
        hidden_state_f = self.init_hidden(batch_size)
        xs_imputation_f = None
        for t in time_step_list_f:
            hidden_state_f, x_c = self.step_f(xs[:, t:t+1], masks[:, t:t+1], d_f[:, t:t+1], hidden_state_f)
            if xs_imputation_f is None:
                xs_imputation_f = x_c
            else:
                xs_imputation_f = torch.cat((xs_imputation_f, x_c), 1)
        h_f = self.fc_f(hidden_state_f)

            # if total_loss is None:
            #     total_loss = x_loss
            # else:
            #     total_loss = total_loss + x_loss
        # return self.fc(hidden_state), x_loss/step_size, xs_imputation

        hidden_state_b = self.init_hidden(batch_size)
        xs_imputation_b = None
        for t in time_step_list_b:
            hidden_state_b, x_cb = self.step_b(xs[:, t:t+1], masks[:, t:t+1], d_f[:, t:t+1], hidden_state_b)
            if xs_imputation_b is None:
                xs_imputation_b = x_cb
            else:
                xs_imputation_b = torch.cat((xs_imputation_b, x_cb), 1)
        h_b = self.fc_b(hidden_state_b)

        return (h_f + h_b)/2, xs_imputation_f, xs_imputation_b

    def step_f(self, x, m, d, h):
        # gamma_f = torch.sigmoid(self.gamma_h_f(d))
        # h = h * gamma_f
        h = h * torch.sigmoid(self.gamma_h_f(d))
        x_h = self.regression_f(h)

        # x_c = m * x + (1 - m) * x_h
        x_c = m * x + (1 - m) * x_h

        # x_loss = torch.sum(torch.abs(x - x_h) * m) / (torch.sum(m) + 1e-5)
        # x_loss = self.x_loss(x*m, x_h*m) / (torch.sum(m) + 1e-5)

        inputs = torch.cat([x_c, m], dim=1)
        # shape of h is (batch_size * hidden_size)
        h = self.rnn_cell_f(inputs, h)
        # return h, x_loss, x_c
        return h, x_h

    def step_b(self, x_b, m_b, d_b, h_b):
        # gamma_b = torch.sigmoid(self.gamma_h_b(d_b))
        # h_b = h_b * gamma_b
        h_b = h_b * torch.sigmoid(self.gamma_h_b(d_b))
        x_hb = self.regression_b(h_b)
        x_cb = m_b * x_b + (1 - m_b) * x_hb
        input_b = torch.cat([x_cb, m_b], dim=1)
        h_b = self.rnn_cell_b(input_b, h_b)
        return h_b, x_hb

    def init_hidden(self, batch_size):

        use_gpu = self.use_gpu
        if use_gpu and self.device == 0:
            hidden_state = Variable(torch.zeros(batch_size, self.rnn_hid_size).cuda())

        elif use_gpu and self.device == 1:
            cuda1 = torch.device('cuda:1')
            hidden_state = Variable(torch.zeros(batch_size, self.rnn_hid_size).cuda(device=cuda1))

        else:
            hidden_state = Variable(torch.zeros(batch_size, self.rnn_hid_size))
        return hidden_state


def train_britsi(model, train_dataloader, test_dataloader, text_path, model_path,
                device=0, num_epochs=40,
                learning_rate_decay=0, learning_rate=0.01, use_gpu=True,
                classes=70, loss_weights=None):
    # create a txt file to record the process of the training
    # a means u can append new txt into the already exiting file and + means u can create a new file if no file exit
    f = open(text_path, 'w+')
    f.write('Model Structure\r\n')
    f.write(str(model) + '\r\n')
    f.close()
    print('Model Structure: ', model)
    print('Start Training ... ')
    if loss_weights is not None:
        loss_weights = loss_weights.astype('float32')
        loss_weights = torch.from_numpy(loss_weights)

    if use_gpu and device == 0:
        print("Let's use GPU 0!")
        if loss_weights is not None:
            loss_weights = Variable(loss_weights.cuda())
        model.cuda()
    if use_gpu and device == 1:
        cuda1 = torch.device('cuda:1')
        print("Let's use GPU 1!")
        if loss_weights is not None:
            loss_weights = Variable(loss_weights.cuda(device=cuda1))
        model.cuda(device=cuda1)
    if not use_gpu:
        print("Let's use CPU")

    criterion = torch.nn.CrossEntropyLoss(weight=loss_weights)
    mseloss = torch.nn.MSELoss(reduction='sum')

    for epoch in range(num_epochs):
        f = open(text_path, 'a+')
        if learning_rate_decay != 0:

            # every [decay_step] epoch reduce the learning rate by half
            if epoch % learning_rate_decay == 0:
                learning_rate = learning_rate / 2
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                f.write('at epoch {} learning_rate is updated to {}\r\n'.format(epoch, learning_rate))
                print('at epoch {} learning_rate is updated to {}'.format(epoch, learning_rate))

        # train the model
        losses, acc = [], []
        label, pred = None, None
        model.train()
        pre_time = time.time()
        # with autograd.detect_anomaly():
        for train_data, train_label in train_dataloader:
            batch_size = train_data.shape[0]
            if use_gpu and device == 0:
                train_data, train_label = Variable(train_data.cuda()), Variable(train_label.cuda())
            if use_gpu and device == 1:
                train_data, train_label = Variable(train_data.cuda(device=cuda1)), Variable(
                    train_label.cuda(device=cuda1))
            else:
                train_data, train_label = Variable(train_data), Variable(train_label)

            optimizer.zero_grad()
            xs = torch.squeeze(train_data[:, 0, :, :])
            xs[xs != xs] = -1
            masks = torch.squeeze(train_data[:, 2, :, :])
            output, x_impu1, x_impu2 = model(train_data)
            x_loss = (mseloss(xs*masks, x_impu1*masks) + mseloss(xs*masks, x_impu2*masks))/ (torch.sum(masks) + 1e-5)\
                     + torch.abs(x_impu1 - x_impu2).mean()*1e-1

            if batch_size == 1:
                y_pred = output.argmax(0)
                pred.append(y_pred)
                label.append(train_label)
                loss = criterion(torch.reshape(output, (1, -1)), train_label) + x_loss
            else:
                y_pred = output.argmax(1)
                if pred is None and label is None:
                    pred = y_pred
                    label = train_label
                else:
                    pred = torch.cat((pred, y_pred), axis=0)
                    label = torch.cat((label, train_label), axis=0)
                loss = criterion(output, train_label) + x_loss
                # loss = criterion(output, train_label)
                # loss = x_loss
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        train_acc = accuracy_score(label.tolist(), pred.tolist())
        train_loss = np.mean(losses)

        # test loss
        losses, acc = [], []
        label, pred = None, None
        model.eval()
        final_y_score = None
        with torch.no_grad():
            for test_data, test_label in test_dataloader:
                if use_gpu and device == 0:
                    test_data, test_label = Variable(test_data.cuda()), Variable(test_label.cuda())
                if use_gpu and device == 1:
                    test_data, test_label = Variable(test_data.cuda(device=cuda1)), Variable(
                        test_label.cuda(device=cuda1))
                else:
                    test_data, test_label = Variable(test_data), Variable(test_label)

                t_xs = torch.squeeze(test_data[:, 0, :, :])
                t_xs[t_xs != t_xs] = -1
                t_masks = torch.squeeze(test_data[:, 2, :, :])
                output, x_1, x_2 = model(test_data)
                test_x_loss = (mseloss(t_xs * t_masks, x_1 * t_masks) + mseloss(t_xs * t_masks, x_2 * t_masks)) / (
                            torch.sum(masks) + 1e-5) \
                         + torch.abs(x_1 - x_2).mean() * 1e-1

                y_score = F.softmax(output, dim=1)

                if final_y_score is None:
                    final_y_score = y_score
                else:
                    final_y_score = torch.cat((final_y_score, y_score), axis=0)

                # Save predict and label
                if test_data.shape[0] == 1:
                    y_pred = output.argmax(0)
                    pred.append(y_pred)
                    label.append(test_label)
                    loss = criterion(torch.reshape(output, (1, -1)), test_label) + test_x_loss
                    # loss = criterion(torch.reshape(y_score, (1, -1)), test_label)
                else:
                    y_pred = output.argmax(1)

                    if pred is None and label is None:
                        pred = y_pred
                        label = test_label
                    else:
                        pred = torch.cat((pred, y_pred), axis=0)
                        label = torch.cat((label, test_label), axis=0)

                    loss = criterion(output, test_label) + test_x_loss
                losses.append(loss.item())

        test_acc = accuracy_score(y_true=label.tolist(), y_pred=pred.tolist())
        test_loss = np.mean(losses)

        test_pred_out = pred
        test_label_out = label
        test_pred_score = label_binarize(test_pred_out.tolist(), classes=list(range(classes)))
        macro_auc1 = roc_auc_score(test_label_out.tolist(), test_pred_score, multi_class='ovo',
                                   labels=list(range(classes)))
        micro_auc1 = roc_auc_score(test_label_out.tolist(), test_pred_score, average='weighted',
                                   multi_class='ovo',
                                   labels=list(range(classes)))
        # macro_auc = roc_auc_score(test_label_out.tolist(), final_y_score.tolist(), multi_class='ovo',
        #                           labels=list(range(classes)))
        # micro_auc = roc_auc_score(test_label_out.tolist(), final_y_score.tolist(), average='weighted',
        #                           multi_class='ovo',
        #                           labels=list(range(classes)))
        macro_p = metrics.precision_score(test_label_out.tolist(), test_pred_out.tolist(), average='macro',
                                          labels=list(range(classes)))
        micro_p = metrics.precision_score(test_label_out.tolist(), test_pred_out.tolist(), average='micro',
                                          labels=list(range(classes)))
        macro_recall = metrics.recall_score(test_label_out.tolist(), test_pred_out.tolist(), average='macro',
                                            labels=list(range(classes)))
        micro_recall = metrics.recall_score(test_label_out.tolist(), test_pred_out.tolist(), average='micro',
                                            labels=list(range(classes)))
        macro_f1 = metrics.f1_score(test_label_out.tolist(), test_pred_out.tolist(), average='macro',
                                    labels=list(range(classes)))
        micro_f1 = metrics.f1_score(test_label_out.tolist(), test_pred_out.tolist(), average='micro',
                                    labels=list(range(classes)))

        training_time = time.time() - pre_time

        a = "Epoch: {} Train loss: {:.4f}, Train acc:{:.4f}, Test loss: {:.4f}, Test acc: {:.4f}, Time: {:.4f}\r\n". \
            format(epoch, train_loss, train_acc, test_loss, test_acc, training_time)
        # b = 'macro auc: {:.4f} and micro auc: {:.4f}\r\n'.format(macro_auc, micro_auc)
        # c = 'zero and one macro auc: {:.4f} and micro auc: {:.4f}\r\n'.format(macro_auc1, micro_auc1)
        c = 'macro auc: {:.4f} and micro auc: {:.4f}\r\n'.format(macro_auc1, micro_auc1)
        d = 'macro precision: {:.4f} and micro precision: {:.4f}\r\n'.format(macro_p, micro_p)
        e = 'macro recall: {:.4f} and micro recall: {:.4f}\r\n'.format(macro_recall, micro_recall)
        g = 'macro f1: {:.4f} and micro f1: {:.4f}\r\n'.format(macro_f1, micro_f1)

        f.write(a + c + d + e + g)
        f.close()
        print(a + c + d + e + g)
        torch.save(model.state_dict(), model_path)


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

    britsi = BRITSI(input_size=1, rnn_hid_size=64, output_size=70, device=1, use_gpu=True)
    train_britsi(britsi, train_loader, test_loader, text_path='result.txt', model_path='result.pt', device=1,
                 num_epochs=100, learning_rate_decay=10, use_gpu=True, loss_weights=normalized_weights)