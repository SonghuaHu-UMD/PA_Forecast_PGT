import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
import datetime
import glob
import matplotlib.dates as mdates
import geopandas as gpd
import os
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GConvGRU, TGCN, A3TGCN, MPNNLSTM, A3TGCN2, DCRNN, TGCN2
from torch_geometric_temporal.nn.attention import MTGNN, STConv, ASTGCN
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric.utils import dense_to_sparse
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.ticker as mtick
from matplotlib.ticker import FuncFormatter


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


DEVICE = torch.device('cuda')  # cuda
shuffle = False
batch_size = 16
d_e = 7 * 2
d_d = 7
d_h = 32
f_s = 2
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


# Get best model location
Metrics_pt = pd.read_csv(r'.\Results\Multi_score_pre_dist_1.csv')
Metrics_pt['Day'] = Metrics_pt['Day'] + 1
M_d = Metrics_pt[Metrics_pt['Day'] == d_d + 1].reset_index(drop=True)
M_d = M_d.loc[M_d.groupby(['model'])['vmse0'].idxmin(), ['train_time', 'vmse0', 'model']]
print(M_d)

# Read best model: batch models T_TGCN2
model = T_TGCN2(batch_size=batch_size, d_h=d_h, d_d=d_d, n_feature=s_feature).to(DEVICE)
best_mpath = r'.\checkpoint\%s_T_TGCN2_model' % M_d.loc[
    M_d['model'] == model.__class__.__name__, 'train_time'].values[0]
test_labels = []
predictions = []
model.load_state_dict(torch.load(best_mpath))
model.eval()
h = None
for encoder_inputs, labels in valid_loader:
    encoder_inputs = encoder_inputs.reshape(batch_size, n_node, s_feature)
    h, y_hat = model(encoder_inputs, static_edge_index, static_edge_weight, h)
    # Store for analysis below
    test_labels.append(labels.cpu().numpy())
    predictions.append(y_hat.detach().cpu().numpy())
f_res = batch_result2df(model, predictions, test_labels, ct_mstd)
f_res.to_pickle(r'.\Results\f_res_%s.pkl' % model.__class__.__name__)

# Read best model: nonbatch models T_GConvGRU
model = T_GConvGRU(in_channels=s_feature, out_channels=d_h, d_d=d_d, filter_size=f_s).to(DEVICE)
best_mpath = r'.\checkpoint\%s_T_GConvGRU_model' % M_d.loc[
    M_d['model'] == model.__class__.__name__, 'train_time'].values[0]
# model = T_DCRNN(in_channels=s_feature, out_channels=d_h, d_d=d_d, filter_size=f_s).to(DEVICE)
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
f_res.to_pickle(r'.\Results\f_res_%s.pkl' % model.__class__.__name__)

# Read best model: nonbatch models T_GConvGRU
model = T_DCRNN(in_channels=s_feature, out_channels=d_h, d_d=d_d, filter_size=f_s).to(DEVICE)
best_mpath = r'.\checkpoint\%s_T_DCRNN_model' % M_d.loc[
    M_d['model'] == model.__class__.__name__, 'train_time'].values[0]
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
f_res.to_pickle(r'.\Results\f_res_%s.pkl' % model.__class__.__name__)

# Read best model: nonbatch models TGCN
model = T_TGCN(in_channels=s_feature, out_channels=d_h, d_d=d_d).to(DEVICE)
best_mpath = r'.\checkpoint\%s_T_TGCN_model' % M_d.loc[
    M_d['model'] == model.__class__.__name__, 'train_time'].values[0]
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
f_res.to_pickle(r'.\Results\f_res_%s.pkl' % model.__class__.__name__)
