import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import seaborn as sns
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GConvGRU, TGCN, A3TGCN, MPNNLSTM, A3TGCN2, DCRNN, TGCN2
from torch_geometric_temporal.nn.attention import MTGNN, STConv, ASTGCN
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric.utils import dense_to_sparse
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from sklearn.metrics import mean_squared_error, mean_absolute_error
import geopandas as gpd


class USDatasetLoader(object):
    def __init__(self, url, A_name, X_name):
        super(USDatasetLoader, self).__init__()
        self._read_web_data(url, A_name, X_name)

    def _read_web_data(self, url, A_name, X_name):
        # url = r"C:\\Users\\huson\\PycharmProjects\\OD_Social_Connect\\data\\"
        A = np.load(os.path.join(url, A_name), allow_pickle=True)
        X = np.load(os.path.join(url, X_name), allow_pickle=True)
        X = X.astype(np.float32)

        # # Normalise as in DCRNN paper (via Z-Score Method)
        # means = np.mean(X, axis=(0, 2))
        # X = X - means.reshape(1, -1, 1)
        # stds = np.std(X, axis=(0, 2))
        # X = X / stds.reshape(1, -1, 1)

        self.A = torch.from_numpy(A)
        self.X = torch.from_numpy(X)

        # return means[0], stds[0]

    def _get_edges_and_weights(self):
        edge_indices, values = dense_to_sparse(self.A)
        edge_indices = edge_indices.numpy()
        values = values.numpy()
        self.edges = edge_indices
        self.edge_weights = values

    def _generate_task(self, num_timesteps_in: int = 12, num_timesteps_out: int = 12):
        """Uses the node features of the graph and generates a feature/target relationship of the shape
        (num_nodes, num_node_features, num_timesteps_in) -> (num_nodes, num_timesteps_out)
        """
        indices = [(i, i + (num_timesteps_in + num_timesteps_out))
                   for i in range(self.X.shape[2] - (num_timesteps_in + num_timesteps_out) + 1)]

        # Generate observations
        features, target = [], []
        for i, j in indices:
            features.append((self.X[:, :, i: i + num_timesteps_in]).numpy())
            target.append((self.X[:, 0, i + num_timesteps_in: j]).numpy())

        self.features = features
        self.targets = target

    def get_dataset(self, num_timesteps_in: int = 12, num_timesteps_out: int = 12) -> StaticGraphTemporalSignal:
        """Returns data iterator as an instance of the static graph temporal signal class."""
        self._get_edges_and_weights()
        self._generate_task(num_timesteps_in, num_timesteps_out)
        dataset = StaticGraphTemporalSignal(self.edges, self.edge_weights, self.features, self.targets)
        return dataset


# Creating DataLoaders
def create_dataloader(train_dataset, DEVICE, bs, shuffle):
    train_input = np.array(train_dataset.features)
    train_target = np.array(train_dataset.targets)
    train_x_tensor = torch.from_numpy(train_input).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
    train_dataset_new = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset_new, batch_size=bs, shuffle=shuffle, drop_last=True)
    return train_loader


# Para setting
DEVICE = torch.device('cuda')  # cuda
shuffle = False
batch_size = 16
d_e = 7 * 2
d_d = 7
d_h = 32
f_s = 2
times = 20
epoches = 40
t_ratio = 0.7
lr = 0.00075

# Read data
url = r".\\data\\"
A_name = "us_adj_mat_dist_1_pre.npy"
X_name = "us_node_values_pre.npy"
loader = USDatasetLoader(url=url, A_name=A_name, X_name=X_name)
dataset = loader.get_dataset(num_timesteps_in=d_e, num_timesteps_out=d_d)
print(next(iter(dataset)))  # Show first sample
A_tilde = torch.from_numpy(np.load(os.path.join(url, A_name), allow_pickle=True)).to(DEVICE)
X_s = torch.from_numpy(np.load(os.path.join(url, "us_node_static_pre.npy"), allow_pickle=True)).to(DEVICE)
ct_mstd = pd.read_pickle(url + r'ct_visit_mstd_pre.pkl')
ct_mstd.rename({'CTFIPS_C': 'id'}, axis=1, inplace=True)

# Test Train Split
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=t_ratio)
valid_dataset, test_dataset = temporal_signal_split(test_dataset, train_ratio=0.5)
print("Number of train buckets: ", len(set(train_dataset)))
print("Number of valid buckets: ", len(set(valid_dataset)))
print("Number of test buckets: ", len(set(test_dataset)))
n_feature = len(dataset.features[0][0])  # Unstack
s_feature = n_feature * len(dataset.features[0][0][0])  # Stack
n_node = len(dataset.features[0])
n_static = X_s.shape[1]

# Creat batches
train_loader = create_dataloader(train_dataset, DEVICE, batch_size, shuffle)
valid_loader = create_dataloader(valid_dataset, DEVICE, batch_size, shuffle)
test_loader = create_dataloader(test_dataset, DEVICE, batch_size, shuffle)

# Static network, get edge features
for snapshot in train_dataset:
    static_edge_index = snapshot.edge_index.to(DEVICE)
    static_edge_weight = snapshot.edge_attr.to(DEVICE)
    break


def mape_loss(y_pred, y):
    return (y - y_pred + 1e-5).abs() / (y + 1e-5).abs()


def smape_loss(y_pred, y):
    return (y - y_pred).abs() / ((y.abs() + y_pred.abs()) / 2)


##### Batch ####
class T_MTGNN(torch.nn.Module):
    def __init__(self, n_node, d_e, d_d, n_feature, n_static, gcn_true, build_adj):
        super(T_MTGNN, self).__init__()
        self.tgnn = MTGNN(gcn_true=gcn_true, build_adj=build_adj, dropout=0, subgraph_size=20, gcn_depth=1,
                          num_nodes=n_node, node_dim=64, dilation_exponential=1, conv_channels=8, residual_channels=8,
                          skip_channels=8, end_channels=16, kernel_size=7, kernel_set=[2, 3, 6, 7], in_dim=n_feature,
                          seq_length=d_e, out_dim=d_d, layers=1, propalpha=0.05, tanhalpha=3, layer_norm_affline=True,
                          xd=n_static)

    def forward(self, x_in, A_tilde, idx=None, FE=None):
        h, A_tildel = self.tgnn(x_in, A_tilde, idx, FE)  # h: (batch_size, num_pred, num of nodes)
        return h[:, :, :, 0].permute(0, 2, 1), A_tildel  # h: (batch_size, num of nodes, num_pred)


class T_MTGNN_NS(torch.nn.Module):
    def __init__(self, n_node, d_e, d_d, n_feature, n_static, gcn_true, build_adj):
        super(T_MTGNN_NS, self).__init__()
        self.tgnn = MTGNN(gcn_true=gcn_true, build_adj=build_adj, dropout=0, subgraph_size=20, gcn_depth=1,
                          num_nodes=n_node, node_dim=64, dilation_exponential=1, conv_channels=8, residual_channels=8,
                          skip_channels=8, end_channels=16, kernel_size=7, kernel_set=[2, 3, 6, 7], in_dim=n_feature,
                          seq_length=d_e, out_dim=d_d, layers=1, propalpha=0.05, tanhalpha=3, layer_norm_affline=True,
                          xd=n_static)

    def forward(self, x_in, A_tilde, idx=None, FE=None):
        h, A_tildel = self.tgnn(x_in, A_tilde, idx, FE)  # h: (batch_size, num_pred, num of nodes)
        return h[:, :, :, 0].permute(0, 2, 1), A_tildel  # h: (batch_size, num of nodes, num_pred)


class T_A3TGCN2(torch.nn.Module):
    def __init__(self, n_feature, d_h, d_d, batch_size):
        super(T_A3TGCN2, self).__init__()
        self.tgnn = A3TGCN2(in_channels=n_feature, out_channels=d_h, periods=d_d, batch_size=batch_size,
                            improved=True, cached=True)
        self.linear = torch.nn.Linear(d_h, d_d)

    def forward(self, x, edge_index, edge_weight, h0):
        h0 = self.tgnn(x, edge_index, edge_weight, h0)  # x [b, n_node, n_feature, d_e] [b, n_node, d_hide]
        y = F.relu(h0)
        y = self.linear(y)
        return h0.detach(), y


class T_TGCN2(torch.nn.Module):
    def __init__(self, n_feature, d_h, d_d, batch_size):
        super(T_TGCN2, self).__init__()
        self.tgnn = TGCN2(in_channels=n_feature, out_channels=d_h, batch_size=batch_size, improved=True, cached=True)
        self.linear = torch.nn.Linear(d_h, d_d)

    def forward(self, x, edge_index, edge_weight, h0):
        h0 = self.tgnn(x, edge_index, edge_weight, h0)
        y = F.relu(h0)
        y = self.linear(y)
        return h0.detach(), y


class T_STConv(torch.nn.Module):
    def __init__(self, n_node, n_feature, d_hide, d_d, kernel_size, d_e):
        super(T_STConv, self).__init__()
        self.recurrent = STConv(num_nodes=n_node, in_channels=n_feature, hidden_channels=d_hide,
                                out_channels=d_hide, kernel_size=kernel_size, K=2, normalization="sym")
        self.linear = torch.nn.Linear(d_hide * (d_e - 2 * (kernel_size - 1)), d_d)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight).to(DEVICE)
        h = F.relu(h).permute(0, 2, 1, 3)
        h = h.reshape(h.shape[0], h.shape[1], -1)
        h = self.linear(h)
        return h


class T_ASTGCN(torch.nn.Module):
    def __init__(self, n_block, n_node, n_feature, d_hide, d_d, d_e, t_str):
        super(T_ASTGCN, self).__init__()
        self.recurrent = ASTGCN(nb_block=n_block, in_channels=n_feature, K=2, nb_chev_filter=d_hide,
                                nb_time_filter=d_hide, time_strides=t_str, num_for_predict=d_d, len_input=d_e,
                                num_of_vertices=n_node, normalization="sym")
        # self.linear = torch.nn.Linear(d_d, d_d).to(DEVICE)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        # h = F.relu(h).to(DEVICE)
        # h = self.linear(h).to(DEVICE)
        return h


##### NON Batch ####
class T_DCRNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, d_d, filter_size):
        super(T_DCRNN, self).__init__()
        self.recurrent = DCRNN(in_channels=in_channels, out_channels=out_channels, K=filter_size)
        self.linear = torch.nn.Linear(out_channels, d_d)

    def forward(self, x, edge_index, edge_weight, h0):
        h0 = self.recurrent(x, edge_index, edge_weight, h0)
        y = F.relu(h0)
        y = self.linear(y)
        return h0.detach(), y


class T_GConvGRU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, d_d, filter_size):
        super(T_GConvGRU, self).__init__()
        self.tgnn = GConvGRU(in_channels=in_channels, out_channels=out_channels, K=filter_size, normalization='sym')
        self.linear = torch.nn.Linear(out_channels, d_d)

    def forward(self, x, edge_index, edge_weight, h0, lambda_max=2):
        h0 = self.tgnn(x, edge_index, edge_weight, h0, lambda_max)
        y = F.relu(h0)
        y = self.linear(y)
        return h0.detach(), y


class T_TGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, d_d):
        super(T_TGCN, self).__init__()
        self.recurrent = TGCN(in_channels=in_channels, out_channels=out_channels, improved=True, cached=True)
        self.linear = torch.nn.Linear(out_channels, d_d)

    def forward(self, x, edge_index, edge_weight, h0):
        h0 = self.recurrent(x, edge_index, edge_weight, h0)
        y = F.relu(h0)
        y = self.linear(y)
        return h0.detach(), y


class T_A3TGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, d_d):
        super(T_A3TGCN, self).__init__()
        self.recurrent = A3TGCN(in_channels=in_channels, out_channels=out_channels, periods=d_d, improved=True,
                                cached=True)
        self.linear = torch.nn.Linear(out_channels, d_d)

    def forward(self, x, edge_index, edge_weight, h0):
        h0 = self.recurrent(x, edge_index, edge_weight, h0)
        y = F.relu(h0)
        y = self.linear(y)
        return h0.detach(), y


class T_MPNNLSTM(torch.nn.Module):
    def __init__(self, in_channels, hidden_size, num_nodes, window, d_d):
        super(T_MPNNLSTM, self).__init__()
        self.recurrent = MPNNLSTM(in_channels=in_channels, hidden_size=hidden_size, num_nodes=num_nodes, window=window,
                                  dropout=0.5)
        self.linear1 = torch.nn.Linear(hidden_size * 2 + in_channels + window - 1, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, d_d)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear1(h)
        h = F.relu(h)
        # h = F.dropout(h, 0.5).to(DEVICE)
        h = self.linear2(h)
        return h


def batch_result2df(model, predictions, test_labels, ct_mstd):
    f_res = pd.DataFrame()
    n_node = predictions[0].shape[1]
    d_d = predictions[0].shape[2]
    batch_size = predictions[0].shape[0]
    for kk in range(0, len(predictions)):
        tf_res = pd.DataFrame(np.concatenate(predictions[kk])).unstack().reset_index()
        tf_res.columns = ['Day', 'Index', 'Predict']
        tf_res['id'] = (list(range(0, n_node)) * batch_size) * d_d
        tf_res['Day_id'] = list(np.repeat(list(range(kk * batch_size, kk * batch_size + batch_size)), n_node)) * d_d
        tf_res['Actual'] = pd.DataFrame(np.concatenate(test_labels[kk])).unstack().reset_index()[0]
        f_res = f_res.append(tf_res)
    r_mse = mean_squared_error(f_res['Predict'], f_res['Actual'])
    print(model.__class__.__name__ + " Valid MSE: {:.4f}".format(r_mse))
    # plt.plot(f_res['Predict'], f_res['Actual'], 'o', alpha=0.01, markersize=1)  # Plot all

    # Re-transform
    f_res = f_res.merge(ct_mstd, on='id', how='left')
    f_res['Predict_r'] = f_res['Predict'] * f_res['std'] + f_res['mean']
    f_res['Actual_r'] = f_res['Actual'] * f_res['std'] + f_res['mean']
    n_mape = mape_loss(f_res['Predict_r'], f_res['Actual_r']).mean()
    print(model.__class__.__name__ + " Valid MAPE: {:.4f}".format(n_mape))

    return f_res


def nonbatch_result2df(model, predictions, test_labels, ct_mstd):
    # Store in DF and plot
    f_res = pd.DataFrame(np.concatenate(predictions)).unstack().reset_index()
    f_res.columns = ['Day', 'Index', 'Predict']
    f_res['id'] = (list(range(0, predictions[0].shape[0])) * (len(predictions))) * predictions[0].shape[1]
    f_res['Day_id'] = list(np.repeat(list(range(0, len(predictions))), predictions[0].shape[0])) * predictions[0].shape[
        1]
    f_res['Actual'] = pd.DataFrame(np.concatenate(test_labels)).unstack().reset_index()[0]
    r_mse = mean_squared_error(f_res['Predict'], f_res['Actual'])
    print(model.__class__.__name__ + " Valid MSE: {:.4f}".format(r_mse))

    # Re-transform
    f_res = f_res.merge(ct_mstd, on='id', how='left')
    f_res['Predict_r'] = f_res['Predict'] * f_res['std'] + f_res['mean']
    f_res['Actual_r'] = f_res['Actual'] * f_res['std'] + f_res['mean']
    f_res['SMAPE'] = smape_loss(f_res['Predict'], f_res['Actual'])
    n_smape = smape_loss(f_res['Predict'], f_res['Actual']).mean()
    print(model.__class__.__name__ + " Valid SMAPE: {:.4f}".format(n_smape))
    return f_res


def cal_metric(f_res):
    vmse, vmae, vmape, vrmse, vmse0, vmae0, vmape0, vrmse0 = [], [], [], [], [], [], [], []
    for i in range(0, f_res['Day'].max() + 1):
        pred = f_res[f_res['Day'] == i]['Predict']
        real = f_res[f_res['Day'] == i]['Actual']
        vmse0.append(mean_squared_error(pred, real))
        vmae0.append(mean_absolute_error(pred, real))
        vmape0.append(mape_loss(pred, real).mean())
        vrmse0.append(mean_squared_error(pred, real, squared=False))
        pred = f_res[f_res['Day'] == i]['Predict_r']
        real = f_res[f_res['Day'] == i]['Actual_r']
        vmse.append(mean_squared_error(pred, real))
        vmae.append(mean_absolute_error(pred, real))
        vmape.append(mape_loss(pred, real).mean())
        vrmse.append(mean_squared_error(pred, real, squared=False))
    vmse0.append(mean_squared_error(f_res['Predict'], f_res['Actual']))
    vmae0.append(mean_absolute_error(f_res['Predict'], f_res['Actual']))
    vmape0.append(mape_loss(f_res['Predict'], f_res['Actual']).mean())
    vrmse0.append(mean_squared_error(f_res['Predict'], f_res['Actual'], squared=False))
    vmse.append(mean_squared_error(f_res['Predict_r'], f_res['Actual_r']))
    vmae.append(mean_absolute_error(f_res['Predict_r'], f_res['Actual_r']))
    vmape.append(mape_loss(f_res['Predict_r'], f_res['Actual_r']).mean())
    vrmse.append(mean_squared_error(f_res['Predict_r'], f_res['Actual_r'], squared=False))
    return vmse0, vmae0, vmape0, vrmse0, vmse, vmae, vmape, vrmse


def train_batch(model, tt, lr, train_loader, valid_loader, test_loader, ct_mstd, batch_size, n_node, s_feature,
                epoches, static_edge_index, static_edge_weight):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 0.00075
    loss_fn = torch.nn.MSELoss()
    total_param = 0
    for param_tensor in model.state_dict():
        # print(param_tensor, '\t', model.state_dict()[param_tensor].size())
        total_param += np.prod(model.state_dict()[param_tensor].size())
    print('Net\'s total params:', total_param)

    # Training and Evaluation
    train_loss, train_eloss, eval_loss, eval_eloss = [], [], [], []
    best_ev = 9999
    for epoch in range(epoches):
        # Train
        model.train()
        at_loss = []
        h = None
        for encoder_inputs, labels in train_loader:
            # print(encoder_inputs.shape)
            encoder_inputs = encoder_inputs.reshape(batch_size, n_node, s_feature)  # (32, 3099, 6*14)
            h, y_hat = model(encoder_inputs, static_edge_index, static_edge_weight, h)
            loss = loss_fn(y_hat, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            at_loss.append(loss.item())
        train_loss.append(at_loss)

        # Evaluation
        with torch.no_grad():
            torch.cuda.empty_cache()
            model.eval()
            ae_loss = []
            h = None
            for encoder_inputs, labels in valid_loader:
                encoder_inputs = encoder_inputs.reshape(batch_size, n_node, s_feature)
                h, y_hat = model(encoder_inputs, static_edge_index, static_edge_weight, h)
                loss = loss_fn(y_hat, labels)
                ae_loss.append(loss.item())
            eval_loss.append(ae_loss)

        print("-- Epoch {} Train MSE: {:.4f} || Valid MSE: {:.4f} --".format(epoch, np.mean(at_loss), np.mean(ae_loss)))

        # Save the model based on evaluation
        if np.mean(eval_loss[epoch]) < best_ev:
            torch.save(model.state_dict(), r'.\checkpoint\%s_%s_model' % (tt, model.__class__.__name__))
            best_mpath = r'.\checkpoint\%s_%s_model' % (tt, model.__class__.__name__)
            best_ev = np.mean(eval_loss[epoch])

    # Evaluation based on best model
    with torch.no_grad():
        torch.cuda.empty_cache()
        test_labels = []
        predictions = []
        model.load_state_dict(torch.load(best_mpath))
        model.eval()
        h = None
        for encoder_inputs, labels in valid_loader:
            encoder_inputs = encoder_inputs.reshape(batch_size, n_node, s_feature)
            h, y_hat = model(encoder_inputs, static_edge_index, static_edge_weight, h)
            test_labels.append(labels.cpu().numpy())
            predictions.append(y_hat.detach().cpu().numpy())

    f_res = batch_result2df(model, predictions, test_labels, ct_mstd)
    vmse0, vmae0, vmape0, vrmse0, vmse, vmae, vmape, vrmse = cal_metric(f_res)

    # Test based on best model
    with torch.no_grad():
        torch.cuda.empty_cache()
        test_labels = []
        predictions = []
        model.load_state_dict(torch.load(best_mpath))
        model.eval()
        h = None
        for encoder_inputs, labels in test_loader:
            encoder_inputs = encoder_inputs.reshape(batch_size, n_node, s_feature)
            h, y_hat = model(encoder_inputs, static_edge_index, static_edge_weight, h)
            test_labels.append(labels.cpu().numpy())
            predictions.append(y_hat.detach().cpu().numpy())

    f_res = batch_result2df(model, predictions, test_labels, ct_mstd)
    mse0, mae0, mape0, rmse0, mse, mae, mape, rmse = cal_metric(f_res)

    return [vmse, vmae, vmape, vrmse, vmse0, vmae0, vmape0, vrmse0, mse, mae, mape, rmse, mse0, mae0, mape0, rmse0]


def train_nonbatch(model, tt, lr, train_dataset, valid_dataset, test_dataset, ct_mstd, n_node, s_feature, epoches,
                   static_edge_index, static_edge_weight):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 0.00075
    loss_fn = torch.nn.MSELoss()
    total_param = 0
    for param_tensor in model.state_dict():
        # print(param_tensor, '\t', model.state_dict()[param_tensor].size())
        total_param += np.prod(model.state_dict()[param_tensor].size())
    print('Net\'s total params:', total_param)

    # Training and Evaluation
    train_loss, eval_loss = [], []
    best_ev = 9999
    for epoch in range(epoches):
        # Train
        model.train()
        at_loss = []
        h = None
        for time, snapshot in enumerate(train_dataset):
            s_node_features_train = torch.FloatTensor(snapshot.x).reshape(n_node, s_feature).to(DEVICE)
            h, y_hat = model(s_node_features_train, static_edge_index, static_edge_weight, h)  # Get model predictions
            loss = loss_fn(y_hat, snapshot.y.to(DEVICE))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            at_loss.append(loss_fn(y_hat, snapshot.y.to(DEVICE)).item())
        train_loss.append(at_loss)

        # Evaluation
        with torch.no_grad():
            torch.cuda.empty_cache()
            model.eval()
            ae_loss = []
            h = None
            for time, snapshot in enumerate(valid_dataset):
                s_node_features_test = torch.FloatTensor(snapshot.x).reshape(n_node, s_feature).to(DEVICE)
                h, y_hat = model(s_node_features_test, static_edge_index, static_edge_weight, h)
                loss = loss_fn(y_hat, snapshot.y.to(DEVICE))
                ae_loss.append(loss.item())
            eval_loss.append(ae_loss)

        print("-- Epoch {} Train MSE: {:.4f} || Valid MSE: {:.4f} --".format(epoch, np.mean(at_loss), np.mean(ae_loss)))

        # Save the model based on evaluation
        if np.mean(eval_loss[epoch]) < best_ev:
            torch.save(model.state_dict(), r'.\checkpoint\%s_%s_model' % (tt, model.__class__.__name__))
            best_mpath = r'.\checkpoint\%s_%s_model' % (tt, model.__class__.__name__)
            best_ev = np.mean(eval_loss[epoch])

    # Evaluation based on best model
    with torch.no_grad():
        torch.cuda.empty_cache()
        test_labels = []
        predictions = []
        model.load_state_dict(torch.load(best_mpath))
        model.eval()
        h = None
        for time, snapshot in enumerate(valid_dataset):
            s_node_features_test = torch.FloatTensor(snapshot.x).reshape(n_node, s_feature).to(DEVICE)
            h, y_hat = model(s_node_features_test, static_edge_index, static_edge_weight, h)
            test_labels.append(snapshot.y.numpy())
            predictions.append(y_hat.detach().cpu().numpy())
    f_res = nonbatch_result2df(model, predictions, test_labels, ct_mstd)
    vmse0, vmae0, vmape0, vrmse0, vmse, vmae, vmape, vrmse = cal_metric(f_res)

    # Test based on best model
    with torch.no_grad():
        torch.cuda.empty_cache()
        test_labels = []
        predictions = []
        model.load_state_dict(torch.load(best_mpath))
        model.eval()
        h = None
        for time, snapshot in enumerate(test_dataset):
            s_node_features_test = torch.FloatTensor(snapshot.x).reshape(n_node, s_feature).to(DEVICE)
            h, y_hat = model(s_node_features_test, static_edge_index, static_edge_weight, h)
            test_labels.append(snapshot.y.numpy())
            predictions.append(y_hat.detach().cpu().numpy())
    f_res = nonbatch_result2df(model, predictions, test_labels, ct_mstd)
    mse0, mae0, mape0, rmse0, mse, mae, mape, rmse = cal_metric(f_res)

    return [vmse, vmae, vmape, vrmse, vmse0, vmae0, vmape0, vrmse0, mse, mae, mape, rmse, mse0, mae0, mape0, rmse0]


# Create each model multiple times
model_mr = pd.DataFrame()
name_col = ['vmse', 'vmae', 'vmape', 'vrmse', 'vmse0', 'vmae0', 'vmape0', 'vrmse0', 'mse', 'mae', 'mape',
            'rmse', 'mse0', 'mae0', 'mape0', 'rmse0']
for tt in range(0, times):
    start = datetime.datetime.now()
    print('____________________%s___________________' % tt)

    model = T_DCRNN(in_channels=s_feature, out_channels=d_h, d_d=d_d, filter_size=f_s).to(DEVICE)
    each_mr = train_nonbatch(model, tt, lr, train_dataset, valid_dataset, test_dataset, ct_mstd, n_node, s_feature,
                             epoches, static_edge_index, static_edge_weight)
    each_mr = pd.DataFrame(each_mr).T
    each_mr.columns = name_col
    each_mr['Day'] = range(0, d_d + 1)
    each_mr['train_time'] = tt
    each_mr['model'] = model.__class__.__name__
    model_mr = model_mr.append(each_mr)

    model = T_GConvGRU(in_channels=s_feature, out_channels=d_h, d_d=d_d, filter_size=f_s).to(DEVICE)
    each_mr = train_nonbatch(model, tt, lr, train_dataset, valid_dataset, test_dataset, ct_mstd, n_node, s_feature,
                             epoches, static_edge_index, static_edge_weight)
    each_mr = pd.DataFrame(each_mr).T
    each_mr.columns = name_col
    each_mr['Day'] = range(0, d_d + 1)
    each_mr['train_time'] = tt
    each_mr['model'] = model.__class__.__name__
    model_mr = model_mr.append(each_mr)

    model = T_TGCN(in_channels=s_feature, out_channels=d_h, d_d=d_d).to(DEVICE)
    each_mr = train_nonbatch(model, tt, lr, train_dataset, valid_dataset, test_dataset, ct_mstd, n_node, s_feature,
                             epoches, static_edge_index, static_edge_weight)
    each_mr = pd.DataFrame(each_mr).T
    each_mr.columns = name_col
    each_mr['Day'] = range(0, d_d + 1)
    each_mr['train_time'] = tt
    each_mr['model'] = model.__class__.__name__
    model_mr = model_mr.append(each_mr)

    # model = T_TGCN2(batch_size=batch_size, d_h=d_h, d_d=d_d, n_feature=s_feature).to(DEVICE)
    # each_mr = train_batch(model, tt, lr, train_loader, valid_loader, test_loader, ct_mstd, batch_size, n_node,
    #                       s_feature, epoches, static_edge_index, static_edge_weight)
    # each_mr = pd.DataFrame(each_mr).T
    # each_mr.columns = name_col
    # each_mr['Day'] = range(0, d_d + 1)
    # each_mr['train_time'] = tt
    # each_mr['model'] = model.__class__.__name__
    # model_mr = model_mr.append(each_mr)

    print(datetime.datetime.now() - start)
model_mr.to_csv(r'.\Results\Multi_score_pre_dist_1.csv')
