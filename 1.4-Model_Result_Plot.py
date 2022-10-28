import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import functools as ft
import datetime
import glob
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
from matplotlib.ticker import FuncFormatter
import geopandas as gpd
import os
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error

pd.options.mode.chained_assignment = None

plt.rcParams.update(
    {'font.size': 13, 'font.family': "serif", 'mathtext.fontset': 'dejavuserif', 'xtick.direction': 'in',
     'xtick.major.size': 0.5, 'grid.linestyle': "--", 'axes.grid': True, "grid.alpha": 1, "grid.color": "#cccccc",
     'xtick.minor.size': 1.5, 'xtick.minor.width': 0.5, 'xtick.minor.visible': True, 'xtick.top': True,
     'ytick.direction': 'in', 'ytick.major.size': 0.5, 'ytick.minor.size': 1.5, 'ytick.minor.width': 0.5,
     'ytick.minor.visible': True, 'ytick.right': True, 'axes.linewidth': 0.5, 'grid.linewidth': 0.5,
     'lines.linewidth': 1.5, 'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05})

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


def mape_loss(y_pred, y):
    return (y - y_pred + 1e-5).abs() / (y + 1e-5).abs()


# Plot boxplot
def plot_box(M_d, hus='model', label_s=['DCRNN', 'GConvGRU', 'TGCN', 'MTGNN'], n_c=5):
    fig, ax = plt.subplots(figsize=(10, 6.5), nrows=2, ncols=2, sharex=True)
    axs = ax.flatten()
    flierprops = dict(markerfacecolor='0.75', markersize=2, linestyle='none')
    sns.boxplot(x='Day', y='vmse', hue=hus, data=M_d, flierprops=flierprops, whis=1.5, ax=axs[0]).set(
        xlabel='Days ahead', ylabel='MSE')
    axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    axs[0].plot([6.5, 6.5], [M_d['vmse'].min(), M_d['vmse'].max()], '--', color='k')

    sns.boxplot(x='Day', y='vmae', hue=hus, data=M_d, flierprops=flierprops, whis=1.5, ax=axs[1]).set(
        xlabel='Days ahead', ylabel='MAE')
    axs[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    axs[1].plot([6.5, 6.5], [M_d['vmae'].min(), M_d['vmae'].max()], '--', color='k')

    sns.boxplot(x='Day', y='vmape', hue=hus, data=M_d, flierprops=flierprops, whis=1.5, ax=axs[2]).set(
        xlabel='Days ahead', ylabel='MAPE')
    axs[2].yaxis.set_major_formatter(FuncFormatter('{0:.0%}'.format))
    axs[2].plot([6.5, 6.5], [M_d['vmape'].min(), M_d['vmape'].max()], '--', color='k')
    axs[2].set_xticks(range(0, 8))
    axs[2].set_xticklabels(['1', ' 2', ' 3', '4', '5', ' 6', ' 7', 'Avg.'])

    sns.boxplot(x='Day', y='vrmse', hue=hus, data=M_d, flierprops=flierprops, whis=1.5, ax=axs[3]).set(
        xlabel='Days ahead', ylabel='RMSE')
    axs[3].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    axs[3].plot([6.5, 6.5], [M_d['vrmse'].min(), M_d['vrmse'].max()], '--', color='k')
    axs[3].set_xticks(range(0, 8))
    axs[3].set_xticklabels(['1', ' 2', ' 3', '4', '5', ' 6', ' 7', 'Avg.'])
    for axss in axs:
        axss.get_legend().remove()
        handles, labels = axss.get_legend_handles_labels()
    fig.legend(handles, label_s, loc='upper right', ncol=n_c, bbox_to_anchor=(0.9, 1))
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.088, right=0.981, hspace=0.13, wspace=0.19)


# Read data
# Get county name
Namecounty = gpd.GeoDataFrame.from_file(
    r'F:\\SafeGraph\\Open Census Data\\Census Website\\2019\\nhgis0011_shape\\US_county_2019.shp')
Namecounty['CTFIPS'] = Namecounty['GEOID']
Namecounty['County_Name'] = Namecounty['NAME'].str.strip().str.title()
STNAME = pd.read_csv(r'D:\OD_Predict\us-state-ansi-fips.csv')
STNAME['STATEFP'] = STNAME[' st'].astype(int).astype(str).apply(lambda x: x.zfill(2))
Namecounty = Namecounty.merge(STNAME, on='STATEFP', how='left')
Namecounty = Namecounty[['CTFIPS', 'County_Name', ' stusps']]
Namecounty['County_Name'] = Namecounty['County_Name'] + ',' + Namecounty[' stusps']

# F1: Plot model metrics by models
sns.set_palette('coolwarm', n_colors=4)
Metrics_pt = pd.read_csv(r'D:\OD_Social_Connect\Results\Multi_score_pre_dist1.csv')
Metrics_gt = pd.read_csv(r'D:\OD_Social_Connect\Results\Multi_score_GTATST.csv')
Metrics_pt = Metrics_pt.append(Metrics_gt)
Metrics_pt['Day'] = Metrics_pt['Day'] + 1
M_d = Metrics_pt[Metrics_pt['Day'] != d_d + 2]
M_d.groupby(['model', 'Day']).mean().reset_index().to_csv(r'D:\OD_Social_Connect\Results\Multi_score_des_mean.csv')
plot_box(M_d, hus='model', label_s=['DCRNN', 'GConvGRU', 'TGCN', 'MTGNN'])
plt.savefig(r'D:\OD_Social_Connect\Results\Model_metrics.png', dpi=1000)

# F2: Different weight matrixes
Metrics_od = pd.read_csv(r'D:\OD_Social_Connect\Results\Multi_score_pre_od.csv')
Metrics_od['weight'] = 'OD'
Metrics_gravity = pd.read_csv(r'D:\OD_Social_Connect\Results\Multi_score_pre_gravity.csv')
Metrics_gravity['weight'] = 'Gravity'
Metrics_fb = pd.read_csv(r'D:\OD_Social_Connect\Results\Multi_score_pre_fb.csv')
Metrics_fb['weight'] = 'Facebook'
Metrics_dist50 = pd.read_csv(r'D:\OD_Social_Connect\Results\Multi_score_pre_dist50.csv')
Metrics_dist50['weight'] = 'Distance'
Metrics_dist1 = pd.read_csv(r'D:\OD_Social_Connect\Results\Multi_score_pre_dist1.csv')
Metrics_dist1['weight'] = 'Self'
M_df = pd.concat([Metrics_od, Metrics_gravity, Metrics_fb, Metrics_dist50, Metrics_dist1], ignore_index=True)
M_df.groupby(['model', 'weight', 'Day']).mean().reset_index().to_csv(
    r'D:\OD_Social_Connect\Results\Multi_weight_des_mean.csv')

M_d = M_df[M_df['model'] == 'T_DCRNN'].reset_index(drop=True)
sns.set_palette('coolwarm', n_colors=5)
plot_box(M_d, hus='weight', label_s=['OD', 'Gravity', 'Facebook', 'Distance', 'Self'])
plt.savefig(r'D:\OD_Social_Connect\Results\DCRNN_metrics.png', dpi=1000)

M_d = M_df[M_df['model'] == 'T_GConvGRU'].reset_index(drop=True)
sns.set_palette('coolwarm', n_colors=5)
plot_box(M_d, hus='weight', label_s=['OD', 'Gravity', 'Facebook', 'Distance', 'Self'])
plt.savefig(r'D:\OD_Social_Connect\Results\GConvGRU_metrics.png', dpi=1000)

M_d = M_df[M_df['model'] == 'T_TGCN'].reset_index(drop=True)
sns.set_palette('coolwarm', n_colors=5)
plot_box(M_d, hus='weight', label_s=['OD', 'Gravity', 'Facebook', 'Distance', 'Self'])
plt.savefig(r'D:\OD_Social_Connect\Results\TGCN_metrics.png', dpi=1000)

# MTGCN
Metrics_od = pd.read_csv(r'D:\OD_Social_Connect\Results\Multi_score_GTAFSF_OD.csv')
Metrics_od['weight'] = 'OD'
Metrics_adp = pd.read_csv(r'D:\OD_Social_Connect\Results\Multi_score_GTATSF.csv')
Metrics_adp['weight'] = 'Adaptive'
Metrics_fb = pd.read_csv(r'D:\OD_Social_Connect\Results\Multi_score_GTAFSF_FB.csv')
Metrics_fb['weight'] = 'Facebook'
Metrics_dist50 = pd.read_csv(r'D:\OD_Social_Connect\Results\Multi_score_GTAFSF_Dis50.csv')
Metrics_dist50['weight'] = 'Distance'
Metrics_dist1 = pd.read_csv(r'D:\OD_Social_Connect\Results\Multi_score_GTAFSF.csv')
Metrics_dist1['weight'] = 'Self'
Metrics_gravity = pd.read_csv(r'D:\OD_Social_Connect\Results\Multi_score_GTAFSF_Gravity.csv')
Metrics_gravity['weight'] = 'Gravity'
M_df = pd.concat([Metrics_od, Metrics_gravity, Metrics_fb, Metrics_dist50, Metrics_adp, Metrics_dist1],
                 ignore_index=True)
M_df.groupby(['model', 'weight', 'Day']).mean().reset_index().to_csv(
    r'D:\OD_Social_Connect\Results\mtgnn_weight_des_mean.csv')

sns.set_palette('coolwarm', n_colors=6)
plot_box(M_df, hus='weight', label_s=['OD', 'Gravity', 'Facebook', 'Distance', 'Self', 'Adaptive'], n_c=6)
plt.savefig(r'D:\OD_Social_Connect\Results\MTGNN_metrics.png', dpi=1000)

# MTGCN
Metrics_od = pd.read_csv(r'D:\OD_Social_Connect\Results\Multi_score_Dur_GFAFSF.csv')
Metrics_od['weight'] = 'F'
Metrics_adp = pd.read_csv(r'D:\OD_Social_Connect\Results\Multi_score_Dur_GTATSF.csv')
Metrics_adp['weight'] = 'T'
Metrics_all = pd.read_csv(r'D:\OD_Social_Connect\Results\Multi_score_Dur_GTATST.csv')
Metrics_all['weight'] = 'AllT'
M_df = pd.concat([Metrics_od, Metrics_adp, Metrics_all], ignore_index=True)
M_df.groupby(['model', 'weight', 'Day']).mean().reset_index().to_csv(
    r'D:\OD_Social_Connect\Results\mtgnn_weight_des_dur_mean.csv')

sns.set_palette('coolwarm', n_colors=3)
plot_box(M_df, hus='weight', label_s=['F', 'T', 'AllT'], n_c=6)

# F3: Plot top and last 4 predictions
plot_list = [r'D:\OD_Social_Connect\Results\f_res_GTATST.pkl', r'D:\OD_Social_Connect\Results\f_res_T_TGCN.pkl',
             r'D:\OD_Social_Connect\Results\f_res_T_DCRNN.pkl', r'D:\OD_Social_Connect\Results\f_res_T_GConvGRU.pkl']
# plot_list = [r'D:\OD_Social_Connect\Results\f_res_GTATST.pkl']
mpl.rcParams['axes.prop_cycle'] = plt.cycler("color", plt.cm.coolwarm(np.linspace(0, 1, 4)))
colors = plt.cm.coolwarm(np.linspace(0, 1, len(plot_list)))
f_res = pd.read_pickle(plot_list[0])
f_res['MAPE'] = mape_loss(f_res['Predict_r'], f_res['Actual_r'])
f_res['MSE'] = (f_res['Predict'] - f_res['Actual']) ** 2
rank_gp = f_res.groupby(['CTFIPS']).mean()['MAPE'].sort_values().reset_index()
top_4 = rank_gp['CTFIPS'][0:4]
last_4 = rank_gp['CTFIPS'][-4:]

fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(12, 6), sharex=True)
cct = 0
dd = 0
axs = ax.flatten()
labels = ['MTGNN', 'TGCN', 'DCRNN', 'GConvGRU']
for kk in plot_list:
    f_res = pd.read_pickle(kk)
    f_res = f_res.merge(Namecounty, on='CTFIPS').reset_index(drop=True)
    ccount = 0
    for idx in top_4:
        temp_test = f_res[(f_res['CTFIPS'] == idx) & ((f_res['Day'] == dd))]
        axs[ccount].plot(temp_test['Day_id'], temp_test['Predict_r'], '-', color=colors[cct],
                         label='Prediction (%s)' % labels[cct], lw=1.5)
        if cct == len(plot_list) - 1:
            axs[ccount].plot(temp_test['Day_id'], temp_test['Actual_r'], '--', label='Actual', color='k', lw=1.5)
        axs[ccount].set_title(list(set(temp_test['County_Name']))[0].split('County')[0], fontsize=12)
        axs[ccount].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        ccount += 1
    for idx in last_4:
        temp_test = f_res[(f_res['CTFIPS'] == idx) & ((f_res['Day'] == dd))]
        axs[ccount].plot(temp_test['Day_id'], temp_test['Predict_r'], '-', color=colors[cct],
                         label='Prediction (%s)' % labels[cct], lw=1.5)
        if cct == len(plot_list) - 1:
            axs[ccount].plot(temp_test['Day_id'], temp_test['Actual_r'], '--', label='Actual', color='k', lw=1.5)
        axs[ccount].set_title(list(set(temp_test['County_Name']))[0].split('County')[0], fontsize=12)
        axs[ccount].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        ccount += 1
    cct += 1
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=11.5)
plt.subplots_adjust(top=0.88, bottom=0.081, left=0.054, right=0.979, hspace=0.152, wspace=0.227)
plt.savefig(r'D:\OD_Social_Connect\Results\TopBottom_all.png', dpi=1000)

# F4: Metrics across county
f_GTATST = pd.read_pickle(r'D:\OD_Social_Connect\Results\f_res_GTATST.pkl')
f_GTATST = f_GTATST[['Day', 'CTFIPS', 'Day_id', 'Predict_r', 'Actual_r']]
f_GTATST.columns = ['Day', 'CTFIPS', 'Day_id', 'Predict_MTGNN', 'Actual_MTGNN']
f_GTATST['MAPE'] = mape_loss(f_GTATST['Predict_MTGNN'], f_GTATST['Actual_MTGNN']) * 100
f_GTATST['MAE'] = (f_GTATST['Predict_MTGNN'] - f_GTATST['Actual_MTGNN']).abs()
Results_df_ct = f_GTATST[f_GTATST['Day'] == 0].groupby('CTFIPS').mean().reset_index()
Results_df_ct.describe().to_csv(r'D:\OD_Social_Connect\Results\Results_ct_dec.csv')

# Read State Geo data
poly_state = gpd.GeoDataFrame.from_file(
    r'F:\\SafeGraph\\Open Census Data\\Census Website\\2019\\nhgis0011_shape\\US_state_2019.shp')
poly_state['STFIPS'] = poly_state['GISJOIN'].str[1:3]
poly_state = poly_state[~poly_state['STFIPS'].isin(['02', '15', '60', '66', '69', '72', '78'])].reset_index(drop=True)
poly_state = poly_state.to_crs(epsg=5070)
poly_raw = gpd.GeoDataFrame.from_file(
    r'F:\\SafeGraph\\Open Census Data\\Census Website\\2019\\nhgis0011_shape\\US_county_2019.shp')
poly_raw['CTFIPS'] = poly_raw['GISJOIN'].str[1:3] + poly_raw['GISJOIN'].str[4:7]
poly_raw = poly_raw.to_crs(epsg=5070)
poly = poly_raw.merge(Results_df_ct, on='CTFIPS')

colormap = 'coolwarm'
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
ax = axs.flatten()
cct = 0
for plot_1 in ['MAPE', 'MAE']:
    poly_state.geometry.boundary.plot(color=None, edgecolor='k', linewidth=1, ax=ax[cct])
    poly.plot(column=plot_1, ax=ax[cct], legend=True, scheme='UserDefined', cmap=colormap,
              classification_kwds=dict(bins=[np.quantile(poly[plot_1], 1 / 6), np.quantile(poly[plot_1], 2 / 6),
                                             np.quantile(poly[plot_1], 3 / 6), np.quantile(poly[plot_1], 4 / 6),
                                             np.quantile(poly[plot_1], 5 / 6)]),
              legend_kwds=dict(frameon=False, ncol=3), linewidth=0, edgecolor='white')
    ax[cct].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    ax[cct].axis('off')
    ax[cct].set_title(plot_1, pad=-20)

    # Reset Legend
    patch_col = ax[cct].get_legend()
    patch_col.set_bbox_to_anchor((0.9, 0.05))
    legend_labels = ax[cct].get_legend().get_texts()
    for bound, legend_label in \
            zip(['< ' + str(round(np.quantile(poly[plot_1], 1 / 6))),
                 str(round(np.quantile(poly[plot_1], 1 / 6))) + ' - ' + str(round(np.quantile(poly[plot_1], 2 / 6))),
                 str(round(np.quantile(poly[plot_1], 2 / 6))) + ' - ' + str(round(np.quantile(poly[plot_1], 3 / 6))),
                 str(round(np.quantile(poly[plot_1], 3 / 6))) + ' - ' + str(round(np.quantile(poly[plot_1], 4 / 6))),
                 str(round(np.quantile(poly[plot_1], 4 / 6))) + ' - ' + str(round(np.quantile(poly[plot_1], 5 / 6))),
                 '> ' + str(round(np.quantile(poly[plot_1], 5 / 6)))], legend_labels):
        legend_label.set_text(bound)
    cct += 1
plt.subplots_adjust(top=0.978, bottom=0.137, left=0.016, right=0.984, hspace=0.2, wspace=0.11)
plt.savefig(r'D:\OD_Social_Connect\Results\Spatial_plot_error_MTGNN.png', dpi=1000)

# F4: Metrics across features
ct_visit = pd.read_pickle(r'D:\OD_Social_Connect\Data\ct_visit.pkl')
ct_visit_one = ct_visit.drop_duplicates(subset=['CTFIPS']).reset_index(drop=True)
n_s = ['Total_Population', 'Median_income', 'ALAND', 'Lng', 'Lat', 'Urbanized_Areas_Population_R',
       'White_Non_Hispanic_R', 'Black_R', 'Asian_R', 'HISPANIC_LATINO_R', 'Bt_18_44_R', 'Bt_45_64_R', 'Over_65_R',
       'Male_R', 'Population_Density', 'Education_Degree_R', 'Unemployed_R', 'No_vehicle_R', 'Democrat_R']
Results_df_ct = Results_df_ct.merge(ct_visit_one[['CTFIPS', 'STFIPS'] + n_s], on='CTFIPS', how='left')
fig, axs = plt.subplots(figsize=(12, 5), ncols=4, nrows=2, sharex=True)  # sharey='row',
ax = axs.flatten()
ccount = 0
for kk in ['MAPE', 'MAE']:
    Results_df_ct['Quantile'] = pd.qcut(Results_df_ct.Total_Population, 10, labels=False)
    sns.boxplot(y=kk, x='Quantile', palette='coolwarm', showfliers=False, ax=ax[ccount * 4 + 0], whis=1.5,
                flierprops=dict(markerfacecolor='0.75', markersize=2, linestyle='none'), data=Results_df_ct)
    ax[ccount * 4 + 0].set_xlabel('')
    ax[ccount * 4 + 0].set_ylabel(kk)

    Results_df_ct['Quantile'] = pd.qcut(Results_df_ct.Black_R, 10, labels=False) + 1
    sns.boxplot(y=kk, x='Quantile', palette='coolwarm', showfliers=False, ax=ax[ccount * 4 + 1], whis=1.5,
                flierprops=dict(markerfacecolor='0.75', markersize=2, linestyle='none'), data=Results_df_ct)
    ax[ccount * 4 + 1].set_xlabel('')
    ax[ccount * 4 + 1].set_ylabel('')

    Results_df_ct['Quantile'] = pd.qcut(Results_df_ct.Median_income, 10, labels=False) + 1
    sns.boxplot(y=kk, x='Quantile', palette='coolwarm', showfliers=False, ax=ax[ccount * 4 + 2], whis=1.5,
                flierprops=dict(markerfacecolor='0.75', markersize=2, linestyle='none'), data=Results_df_ct)
    ax[ccount * 4 + 2].set_xlabel('')
    ax[ccount * 4 + 2].set_ylabel('')

    Results_df_ct['Quantile'] = pd.qcut(Results_df_ct.Democrat_R, 10, labels=False) + 1
    sns.boxplot(y=kk, x='Quantile', palette='coolwarm', showfliers=False, ax=ax[ccount * 4 + 3], whis=1.5,
                flierprops=dict(markerfacecolor='0.75', markersize=2, linestyle='none'), data=Results_df_ct)
    ax[ccount * 4 + 3].set_xlabel('')
    ax[ccount * 4 + 3].set_ylabel('')

    ax[1 * 4 + 0].set_xlabel('Population')
    ax[1 * 4 + 1].set_xlabel('African American')
    ax[1 * 4 + 2].set_xlabel('Median Income')
    ax[1 * 4 + 3].set_xlabel('Democrat')

    if kk in ['MAE', 'RMSE']:
        ax[ccount * 4 + 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        ax[ccount * 4 + 1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        ax[ccount * 4 + 2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        ax[ccount * 4 + 3].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)

    ccount += 1
    plt.subplots_adjust(top=0.981, bottom=0.1, left=0.068, right=0.984, hspace=0.187, wspace=0.23)
plt.savefig('D:\OD_Social_Connect\Results\Metrics_by_Variables_MTGNN.png', dpi=1200)
# plt.close()

# F5: Model learned spatial relationship
poly_raw = gpd.GeoDataFrame.from_file(
    r'F:\\SafeGraph\\Open Census Data\\Census Website\\2019\\nhgis0011_shape\\US_county_2019.shp')
poly_raw['CTFIPS'] = poly_raw['GISJOIN'].str[1:3] + poly_raw['GISJOIN'].str[4:7]
poly_raw = poly_raw.to_crs(epsg=4326)
ct_mstd = pd.read_pickle(r'D:\OD_Social_Connect\Data\ct_visit_mstd_pre.pkl')
ct_mstd.rename({'CTFIPS_C': 'id'}, axis=1, inplace=True)

adp_pd = pd.read_pickle(r'C:\Users\huson\PycharmProjects\OD_Social_Connect\MTGNN-OD\save\Dur_GTATST1_0_weight.pkl')
county_plot = adp_pd[(adp_pd[0] != 0)].reset_index(drop=True)
county_plot.columns = ['O_CTFIPS_C', 'D_CTFIPS_C', 'weight']
county_plot['weight'].describe()

# Agg
county_agg = county_plot.groupby(['O_CTFIPS_C']).sum()['weight'].reset_index()
ct_mstd_ll = ct_mstd[['id', 'CTFIPS']]
ct_mstd_ll.columns = ['O_CTFIPS_C', 'CTFIPS']
ct_mstd_ll = ct_mstd_ll.merge(Namecounty, on='CTFIPS').reset_index(drop=True)
county_agg = county_agg.merge(ct_mstd_ll, on='O_CTFIPS_C').sort_values(by='weight').reset_index(drop=True)

county_plot = county_plot[(county_plot['weight'] > np.percentile(county_plot['weight'], 99.5))]
print(len(county_plot))
ct_mstd_ll = ct_mstd[['id', 'Lng', 'Lat']]
ct_mstd_ll.columns = ['O_CTFIPS_C', 'O_Lng', 'O_Lat']
county_plot = county_plot.merge(ct_mstd_ll, on='O_CTFIPS_C')
ct_mstd_ll.columns = ['D_CTFIPS_C', 'D_Lng', 'D_Lat']
county_plot = county_plot.merge(ct_mstd_ll, on='D_CTFIPS_C').reset_index(drop=True)

poly = poly_raw.merge(ct_mstd, on='CTFIPS')
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(9, 6))
poly.geometry.boundary.plot(color=None, edgecolor='gray', linewidth=0.2, ax=ax)
for kk in range(0, len(county_plot)):
    ax.annotate('', xy=(county_plot.loc[kk, 'O_Lng'], county_plot.loc[kk, 'O_Lat']),
                xytext=(county_plot.loc[kk, 'D_Lng'], county_plot.loc[kk, 'D_Lat']),
                arrowprops={'arrowstyle': '-', 'lw': county_plot.loc[kk, 'weight'], 'color': 'blue', 'alpha': 0.1,
                            'connectionstyle': "arc3,rad=0.2"}, va='center')
ax.axis('off')
plt.tight_layout()
plt.savefig(r'D:\OD_Social_Connect\Results\Weight_Learned.png', dpi=1000)
plt.close()
