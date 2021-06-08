from models.MLP_GRU.MLP_GRU import MLP_GRU
from data.sparse_data_generating import *
import torch
from torch.autograd import Variable
from sklearn.metrics import accuracy_score


def sparse_gra():
    # load testing data and labels
    df_com_data = pd.read_csv('../../data/complete_data.csv')
    total_rows = df_com_data.shape[0]
    p = pd.read_csv('../../data/sparse_distribution.csv').iloc[:,0].values
    sub_df_com_data = sub_dataframe(df_com_data.iloc[round(0.8 * total_rows):, :])
    df_label1 = pd.read_csv('../../data/complete_data_labels1.csv')
    coarse_labels = df_label1.iloc[round(0.8 * total_rows):, :].values
    df_label2 = pd.read_csv('../../data/complete_data_labels2.csv')
    medium_labels = df_label2.iloc[round(0.8 * total_rows):, :].values
    df_label3 = pd.read_csv('../../data/complete_data_labels3.csv')
    fine_labels = df_label3.iloc[round(0.8 * total_rows):, :].values

    testing_sparsity = [5,10,15,20,25,30,35,40,45,50,55,60]
    for n in testing_sparsity:
        # prepare testing data
        sparse_data = down_sampling(sub_df_com_data, n=288-n, p=p, back_flag=True)
        test_x = sparse_data.astype('float32')
        total_number = test_x.shape[0]
        type_size = test_x.shape[1]
        time_step = test_x.shape[2]
        test_x = test_x.reshape(total_number, type_size, time_step, -1)
        test_data = torch.from_numpy(test_x)
        test_data = Variable(test_data)

        # load models
        mlpgru_fine = MLP_GRU(input_size=1, gru_hidden_size=64, mlp_hidden_size=256, mlp_layer=1, output_size=70,
                        device=0, output_last=True, use_gpu=False)
        mlpgru_fine.load_state_dict(torch.load('mlp_gru3.pt', map_location="cpu"))

        mlpgru_medium = MLP_GRU(input_size=1, gru_hidden_size=64, mlp_hidden_size=256, mlp_layer=1, output_size=15,
                              device=0, output_last=True, use_gpu=False)
        mlpgru_medium.load_state_dict(torch.load('mlp_gru2.pt', map_location="cpu"))

        mlpgru_coarse = MLP_GRU(input_size=1, gru_hidden_size=64, mlp_hidden_size=256, mlp_layer=1, output_size=5,
                              device=0, output_last=True, use_gpu=False)
        mlpgru_coarse.load_state_dict(torch.load('mlp_gru1.pt', map_location="cpu"))

        fine_predicted = mlpgru_fine(test_data).argmax(1)
        acc_fine = accuracy_score(fine_labels, fine_predicted)

        medium_predicted = mlpgru_medium(test_data).argmax(1)
        acc_medium = accuracy_score(medium_labels, medium_predicted)

        coarse_predicted = mlpgru_coarse(test_data).argmax(1)
        acc_coarse = accuracy_score(coarse_labels, coarse_predicted)

        sparsity = (288-n)/288
        print("when sparsity is {:.2%}".format(sparsity))
        print("average accuracy of mlp_gru on fine granularity labels is {}".format(acc_fine))
        print("average accuracy of mlp_gru on medium granularity labels is {}".format(acc_medium))
        print("average accuracy of mlp_gru on coarse granularity labels is {}".format(acc_coarse))


if __name__ == "__main__":
    sparse_gra()