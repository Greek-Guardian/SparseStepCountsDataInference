"""
prepare data to train and evaluate model
"""
import pandas as pd
import numpy as np
import os
from collections import Counter
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
import time
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.preprocessing import label_binarize


def prepare_mixed_data_granularity(root_path, granularity):
    """
    prepare mixed training dataset and labels
    :param root_path:
    :param granularity: 1,2,3
    :return:
    """
    # load data and labels
    df_labels = pd.read_csv(os.path.join(root_path, 'complete_data_labels{}.csv'.format(granularity)))
    test_labels = df_labels.iloc[40000:, 0].values
    train_labels = df_labels.iloc[0:40000, 0].values
    final_train_labels = train_labels
    for i in range(5):
        final_train_labels = np.append(final_train_labels, train_labels, axis=0)
    # print(final_train_labels.shape)

    traindata_names = ["single_all_sparse510.npy", "single_all_sparse1020.npy", "single_all_sparse2050.npy",
                       "single_all_sparse50100.npy", "single_all_sparse100150.npy", "single_all_sparse278283.npy"]
    train_x = np.array([])
    for name in traindata_names:
        single_sparse_data = np.load(os.path.join(root_path, name))
        train_data = single_sparse_data[0:40000, :, :]
        if name == "single_all_sparse1020.npy":
            test_x = single_sparse_data[40000:, :, :]
        if train_x.shape[0] == 0:
            train_x = train_data
        else:
            train_x = np.vstack([train_x, train_data])
        # print(train_x)


    train_x = train_x.astype('float32')
    test_x = test_x.astype('float32')
    return final_train_labels, test_labels, train_x, test_x


def prepare_train_test_data(train_data, test_data, train_labels, test_labels, batch_size=50, reshape_flag=True):
    """
    prepare train and test data loader respectively
    :param train_data: np.arrays
    :param test_data: np.arrays
    :param train_labels: np.arrays
    :param test_labels: np.arrays
    :param batch_size:
    :return:
    """
    train_num = train_data.shape[0]
    test_num = test_data.shape[0]
    type_size = test_data.shape[1]
    time_steps = test_data.shape[2]
    if reshape_flag:
        train_data = train_data.reshape(train_num, type_size, time_steps, -1)
        test_data = test_data.reshape(test_num, type_size, time_steps, -1)

    train_dataset = TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_labels).squeeze())
    test_dataset = TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_labels).squeeze())
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

    return train_dataloader, test_dataloader


def train_model(model, train_dataloader, test_dataloader, text_path, model_path,
                device=0, target_replication=False, alpha=0.5,
                num_epochs=40,
                learning_rate_decay=0, learning_rate=0.01, use_gpu=True,
                classes=70, loss_weights=None):
    """
    train classification model
    :param model: RNN models
    :param train_dataloader:
    :param test_dataloader:
    :param text_path: store path of result
    :param model_path: store path of model
    :param device: 0 or 1, which means GPU0 or GPU1
    :param target_replication: whether to use history hidden state to predict labels
    :param alpha: the weighting coefficient of last hidden state
    :param num_epochs: training epochs
    :param learning_rate_decay: the number of epochs to decay learning rate
    :param learning_rate:
    :param use_gpu: true or false
    :param classes: the total number of classes
    :param loss_weights: weights assigned to each class
    :return:
    """
    # create a txt file to record the process of the training
    # a means u can append new txt into the already exiting file and + means u can create a new file if no file exit
    f = open(text_path, 'w+')
    f.write('Model Structure\r\n')
    f.write(str(model)+'\r\n')
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

    # if (type(model) == nn.modules.container.Sequential):
    #     output_last = model[-1].output_last
    #     print('Output type dermined by the last layer')
    # else:
    #     output_last = model.output_last
    #     print('Output type dermined by the model')

    criterion = torch.nn.CrossEntropyLoss(weight=loss_weights)
    epoch_losses = []

    # Variables for Early Stopping
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
            output = model(train_data)

            if not target_replication:
                weighted_output = output

            if target_replication and alpha > 1:
                time_steps = output.shape[1]
                time_sum = (1 + 288) * 144
                weighted_output = None
                for i in range(time_steps):
                    if weighted_output is None:
                        weighted_output = (i + 1) / time_sum * output[:, i, :]
                    else:
                        weighted_output += (i + 1) / time_sum * output[:, i, :]

            if target_replication and 0 < alpha < 1:
                time_steps = output.shape[1]
                weighted_output = (1 - alpha) * output[:, -1, :]
                for i in range(time_steps):
                    weighted_output += (alpha / time_steps) * output[:, i, :]

            if batch_size == 1:
                y_pred = output.argmax(0)
                pred.append(y_pred)
                label.append(train_label)
                loss = criterion(torch.reshape(output, (1, -1)), train_label)
                # loss = criterion(torch.reshape(y_score, (1, -1)), train_label)
            else:
                y_pred = weighted_output.argmax(1)

                if pred is None and label is None:
                    pred = y_pred
                    label = train_label
                else:
                    pred = torch.cat((pred, y_pred), axis=0)
                    label = torch.cat((label, train_label), axis=0)

                if not target_replication:
                     loss = criterion(output, train_label)

                if target_replication and alpha > 1:
                    time_steps = output.shape[1]
                    time_sum = (1+288)*144
                    loss = None
                    for i in range(time_steps):
                        each_loss = criterion(output[:, i, :], train_label)
                        if loss is None:
                            loss = (i+1)/time_sum * each_loss
                        else:
                            loss += (i+1)/time_sum * each_loss

                if target_replication and 0<alpha<1:
                    time_steps = output.shape[1]
                    last_loss = (1-alpha)*criterion(output[:, -1, :], train_label)
                    for i in range(time_steps):
                        each_loss = criterion(output[:, i, :], train_label)
                        loss = last_loss + (alpha/time_steps)*each_loss
                # loss = criterion(y_score, train_label)
            losses.append(loss.item())
            # losses.append(loss.detach())
            # perform a backward pass, and update the weights.
            loss.backward()
            optimizer.step()

        train_acc = accuracy_score(label.tolist(), pred.tolist())
        train_loss = np.mean(losses)

        train_pred_out = pred
        train_label_out = label

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
                    test_data, test_label = Variable(test_data.cuda(device=cuda1)), Variable(test_label.cuda(device=cuda1))
                else:
                    test_data, test_label = Variable(test_data), Variable(test_label)

                output = model(test_data)

                if not target_replication:
                    weighted_output = output

                if target_replication and alpha > 1:
                    time_steps = output.shape[1]
                    time_sum = (1 + 288) * 144
                    weighted_output = None
                    for i in range(time_steps):
                        if weighted_output is None:
                            weighted_output = (i + 1) / time_sum * output[:, i, :]
                        else:
                            weighted_output += (i + 1) / time_sum * output[:, i, :]

                if target_replication and 0 < alpha < 1:
                    time_steps = output.shape[1]
                    weighted_output = (1 - alpha) * output[:, -1, :]
                    for i in range(time_steps):
                        weighted_output += (alpha / time_steps) * output[:, i, :]

                y_score = F.softmax(weighted_output, dim=1)

                if final_y_score is None:
                    final_y_score = y_score
                else:
                    final_y_score = torch.cat((final_y_score, y_score), axis=0)

                # Save predict and label
                if test_data.shape[0] == 1:
                    y_pred = output.argmax(0)
                    pred.append(y_pred)
                    label.append(test_label)
                    loss = criterion(torch.reshape(output, (1, -1)), test_label)
                    # loss = criterion(torch.reshape(y_score, (1, -1)), test_label)
                else:
                    y_pred = weighted_output.argmax(1)

                    if pred is None and label is None:
                        pred = y_pred
                        label = test_label
                    else:
                        pred = torch.cat((pred, y_pred), axis=0)
                        label = torch.cat((label, test_label), axis=0)

                    loss = criterion(weighted_output, test_label)
                    #     last_loss = (1 - alpha) * criterion(output[:, -1, :], test_label)
                    #     for i in range(time_steps):
                    #         each_loss = criterion(output[:, i, :], test_label)
                    #         loss = last_loss + (alpha / time_steps) * each_loss

                    # loss = criterion(y_score, test_label)
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

        a = "Epoch: {} Train loss: {:.4f}, Train acc:{:.4f}, Test loss: {:.4f}, Test acc: {:.4f}, Time: {:.4f}\r\n".\
            format(epoch, train_loss,train_acc, test_loss, test_acc, training_time)

        b = 'macro auc: {:.4f} and micro auc: {:.4f}\r\n'.format(macro_auc1, micro_auc1)
        c = 'macro precision: {:.4f} and micro precision: {:.4f}\r\n'.format(macro_p, micro_p)
        d = 'macro recall: {:.4f} and micro recall: {:.4f}\r\n'.format(macro_recall, micro_recall)
        e = 'macro f1: {:.4f} and micro f1: {:.4f}\r\n'.format(macro_f1, micro_f1)

        f.write(a+b+c+d+e)
        f.close()
        print(a+b+c+d+e)
        torch.save(model.state_dict(), model_path)
        # print("test predict:", test_pred_out[0:100])
        # print("true test labels:", test_label_out[0:100])
    # return test_pred_out, test_label_out


def balance_loss_weight(labels_list, total_labels, beta):
    """
    output weights based on paper "class-balanced loss based on ffective number of samples" (CVPR2019)
    :param labels_list: labels of all samples
    :param total_labels: the number of class categories
    :param beta: hyperparameter
    :return:
    """
    c = Counter(labels_list)
    number_per_class = [-1] * total_labels
    for k, v in c.items():
        if v == 0:
            raise ValueError("class {} has no labels".format(k))
        else:
            number_per_class[k] = v

    effective_num = 1.0 - np.power(beta, number_per_class)
    weights = (1.0 - beta) / np.array(effective_num)
    normalized_weights = weights / np.sum(weights) * total_labels
    return normalized_weights