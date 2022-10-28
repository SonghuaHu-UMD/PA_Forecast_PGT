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

pd.options.mode.chained_assignment = None

plt.rcParams.update(
    {'font.size': 13, 'font.family': "serif", 'mathtext.fontset': 'dejavuserif', 'xtick.direction': 'in',
     'xtick.major.size': 0.5, 'grid.linestyle': "--", 'axes.grid': True, "grid.alpha": 1, "grid.color": "#cccccc",
     'xtick.minor.size': 1.5, 'xtick.minor.width': 0.5, 'xtick.minor.visible': True, 'xtick.top': True,
     'ytick.direction': 'in', 'ytick.major.size': 0.5, 'ytick.minor.size': 1.5, 'ytick.minor.width': 0.5,
     'ytick.minor.visible': True, 'ytick.right': True, 'axes.linewidth': 0.5, 'grid.linewidth': 0.5,
     'lines.linewidth': 1.5, 'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05})


# Distance
def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h


# Read data
ct_visit = pd.read_pickle(r'D:\OD_Social_Connect\Data\ct_visit.pkl')
# datetime.datetime(2020, 6, 1)
ct_visit = ct_visit[ct_visit['Date'] < datetime.datetime(2020, 2, 1)].reset_index(drop=True)
# ct_visit = ct_visit[(ct_visit['Date'] > datetime.datetime(2019, 1, 1)) & (
#             ct_visit['Date'] < datetime.datetime(2021, 10, 1))].reset_index(drop=True)
t_start = ct_visit['Date'].min()
t_end = ct_visit['Date'].max()
train_ratio = 0.7
val_ratio = 0.15
split_time = t_start + datetime.timedelta(days=int(((t_end - t_start).total_seconds() / (24 * 3600)) * train_ratio))
test_time = t_start + datetime.timedelta(
    days=int(((t_end - t_start).total_seconds() / (24 * 3600)) * (train_ratio + val_ratio)))
print(split_time)
print(test_time)

# Read county shp
poly_raw = gpd.GeoDataFrame.from_file(
    r'F:\\SafeGraph\\Open Census Data\\Census Website\\2019\\nhgis0011_shape\\US_county_2019.shp')
poly_raw['CTFIPS'] = poly_raw['GISJOIN'].str[1:3] + poly_raw['GISJOIN'].str[4:7]
poly_raw = poly_raw.to_crs(epsg=4326)

# Drop those small CT
ct_visit_mean = ct_visit.groupby(['CTFIPS'])['Daily_Flow'].mean().reset_index()
need_ct = ct_visit_mean.loc[ct_visit_mean['Daily_Flow'] >= 100, 'CTFIPS']
ct_visit = ct_visit.merge(need_ct, on='CTFIPS').reset_index(drop=True)
# Drop some outliers
ct_visit = ct_visit[~ct_visit['CTFIPS'].isin(['47005', '21077', '51157', '51013', '08023'])].reset_index(drop=True)
print(len(set(ct_visit['CTFIPS'])))

# CTFIPS encode
ct_visit["CTFIPS"] = ct_visit["CTFIPS"].astype('category')
ct_visit["CTFIPS_C"] = ct_visit["CTFIPS"].cat.codes

## Group normalize Y based on training dataset
ct_visit_train = ct_visit[ct_visit['Date'] <= split_time].reset_index(drop=True)
ct_visit_mean = ct_visit_train.groupby(['CTFIPS_C'])['Daily_Flow'].mean().reset_index()
ct_visit_std = ct_visit_train.groupby(['CTFIPS_C'])['Daily_Flow'].std().reset_index()
ct_visit_mstd = ct_visit_mean.merge(ct_visit_std, on='CTFIPS_C')
ct_visit_mstd.columns = ['CTFIPS_C', 'mean', 'std']
ct_visit = ct_visit.merge(ct_visit_mstd, on='CTFIPS_C')
ct_visit['Daily_Flow_Raw'] = ct_visit['Daily_Flow']
ct_visit['Daily_Flow'] = (ct_visit['Daily_Flow'] - ct_visit['mean']) / ct_visit['std']
# ct_visit[ct_visit['Daily_Flow'] > 10]
# Store mean and std
ct_visit.drop_duplicates(subset=['CTFIPS']).reset_index(drop=True)[
    ['CTFIPS', 'CTFIPS_C', 'mean', 'std', 'Lng', 'Lat']].to_pickle(r'D:\OD_Social_Connect\Data\ct_visit_mstd_pre.pkl')

# Figure 1: Daily plot, normalize
mpl.rcParams['axes.prop_cycle'] = plt.cycler("color", plt.cm.coolwarm(np.linspace(0, 1, 100)))
fig, ax = plt.subplots(figsize=(10, 6))
for kk in set(ct_visit['CTFIPS']):
    tempfile = ct_visit[ct_visit['CTFIPS'] == kk]
    ax.plot(tempfile['Date'], tempfile['Daily_Flow'], label=kk, alpha=0.4, lw=1)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.set_ylabel('Normalized transformation')
ax.set_xlabel('Date')
ax.plot(ct_visit.groupby('Date')['Daily_Flow'].mean(), color='k', alpha=0.6, lw=2)
ax.plot([split_time, split_time], [-6, 13], '-.', color='green', alpha=0.6, lw=3)
ax.plot([test_time, test_time], [-6, 13], '-.', color='blue', alpha=0.6, lw=3)
plt.tight_layout()
plt.savefig(r'D:\OD_Social_Connect\Results\Normal_daily.png', dpi=1000)
plt.close()

# # Figure 1: Daily plot, nonnormalize, by county
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

mpl.rcParams['axes.prop_cycle'] = plt.cycler("color", plt.cm.coolwarm(np.linspace(0, 1, 4)))
rank_gp = ct_visit.groupby(['CTFIPS']).median()['Daily_Flow_Raw'].sort_values().reset_index()
top_4 = rank_gp['CTFIPS'][0:4]
last_4 = rank_gp['CTFIPS'][-4:]

fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(12, 6), sharex=True)
axs = ax.flatten()
f_res = ct_visit.merge(Namecounty, on='CTFIPS').reset_index(drop=True)
ccount = 0
for idx in last_4:
    temp_test = f_res[(f_res['CTFIPS'] == idx)]
    axs[ccount].plot(temp_test['Date'], temp_test['Daily_Flow_Raw'], '-', lw=1.5)
    axs[ccount].set_title(list(set(temp_test['County_Name']))[0].split('County')[0], fontsize=12)
    axs[ccount].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    axs[ccount].xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
    axs[ccount].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ccount += 1
for idx in top_4:
    temp_test = f_res[(f_res['CTFIPS'] == idx)]
    axs[ccount].plot(temp_test['Date'], temp_test['Daily_Flow_Raw'], '-', lw=1.5)
    axs[ccount].set_title(list(set(temp_test['County_Name']))[0].split('County')[0], fontsize=12)
    axs[ccount].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    axs[ccount].xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
    axs[ccount].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ccount += 1
plt.tight_layout()
plt.savefig(r'D:\OD_Social_Connect\Results\TopBottom_Origin.png', dpi=1000)

## Normalize time-varying X based on training dataset
n_x = ['Is_holiday', 'dayofweek', 'Isweekend', 'TEMP', 'PRCP']
for kk in n_x: ct_visit[kk] = (ct_visit[kk] - ct_visit_train[kk].mean()) / ct_visit_train[kk].std()

# Output time-varying X
X_c = np.array(list(ct_visit[['CTFIPS', 'Daily_Flow'] + n_x].groupby('CTFIPS').apply(pd.DataFrame.to_numpy)))
X_c = np.transpose(X_c, (0, 2, 1))
X_c = np.delete(X_c, 0, axis=1)  # (3097, 6, 854)
np.save(r'C:\Users\huson\PycharmProjects\OD_Social_Connect\data\us_node_values_pre.npy', X_c)

## Normalize static X
ct_visit_one = ct_visit.drop_duplicates(subset=['CTFIPS']).reset_index(drop=True)
n_s = ['Total_Population', 'Median_income', 'ALAND', 'Lng', 'Lat', 'Urbanized_Areas_Population_R',
       'White_Non_Hispanic_R', 'Black_R', 'Asian_R', 'HISPANIC_LATINO_R', 'Bt_18_44_R', 'Bt_45_64_R', 'Over_65_R',
       'Male_R', 'Population_Density', 'Education_Degree_R', 'Unemployed_R', 'No_vehicle_R', 'Democrat_R']
for kk in n_s: ct_visit_one[kk] = (ct_visit_one[kk] - ct_visit_one[kk].mean()) / ct_visit_one[kk].std()

# Output static X
ct_visit_one[n_s] = ct_visit_one[n_s].astype(np.float32)
X_s = np.array(list(ct_visit_one[n_s].to_numpy()))  # (3097, 45) (3097, 19)
np.save(r'C:\Users\huson\PycharmProjects\OD_Social_Connect\data\us_node_static_pre.npy', X_s)

#### Calculate edge weight
county_ll = ct_visit.drop_duplicates(subset=['CTFIPS']).reset_index(drop=True)[['CTFIPS', 'Lng', 'Lat']]
county_ll[['Lng', 'Lat']] = county_ll[['Lng', 'Lat']].astype(float)
poly = poly_raw.merge(county_ll, on='CTFIPS')
# Method 1: Pairwise distance
county_re = pd.DataFrame(np.repeat(county_ll.to_numpy(), len(county_ll), axis=0), columns=county_ll.columns)
county_re.columns = ['O_CTFIPS', 'O_Lng', 'O_Lat']
county_re_1 = county_ll.iloc[np.tile(np.arange(len(county_ll)), len(county_ll))].reset_index(drop=True)
county_re_1.columns = ['D_CTFIPS', 'D_Lng', 'D_Lat']
county_re = pd.concat([county_re, county_re_1], axis=1)
del county_re_1
county_re[['O_Lng', 'O_Lat', 'D_Lng', 'D_Lat']] = county_re[['O_Lng', 'O_Lat', 'D_Lng', 'D_Lat']].astype(float)
county_re['Dist'] = haversine_array(county_re['O_Lat'], county_re['O_Lng'], county_re['D_Lat'], county_re['D_Lng']) + 1

# t_h as threshold, calculate weight
t_h = 50  # km
print('No of edges: %s' % (len(county_re[county_re['Dist'] <= t_h])))
county_re['Dist_log'] = 1 / np.log(county_re['Dist'] + 1)
county_re['Dist_exp'] = np.exp(-county_re['Dist'] ** 2 / np.mean(county_re[county_re['Dist'] <= t_h]['Dist']) ** 2)
county_re['weight'] = 0
county_re.loc[county_re['Dist'] <= t_h, 'weight'] = county_re.loc[county_re['Dist'] <= t_h, 'Dist_exp']
county_re.loc[county_re['O_CTFIPS'] == county_re['D_CTFIPS'], 'weight'] = 1
# sns.displot(county_re[county_re['weight'] != 0]['weight'])

# County static feature to array and output
county_dist_wide = county_re.pivot(index='O_CTFIPS', columns='D_CTFIPS', values='weight')
county_dist_wide = county_dist_wide.astype(np.float32)
A_c = county_dist_wide.to_numpy()
np.save(r'C:\Users\huson\PycharmProjects\OD_Social_Connect\data\us_adj_mat_dist_50_pre.npy', A_c)

# Plot graph
county_plot = county_re[(county_re['weight'] != 0) & (county_re['weight'] < county_re['weight'].max())].reset_index(
    drop=True)
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(9, 6))
poly.geometry.boundary.plot(color=None, edgecolor='gray', linewidth=0.2, ax=ax)
for kk in range(0, len(county_plot)):
    ax.annotate('', xy=(county_plot.loc[kk, 'O_Lng'], county_plot.loc[kk, 'O_Lat']),
                xytext=(county_plot.loc[kk, 'D_Lng'], county_plot.loc[kk, 'D_Lat']),
                arrowprops={'arrowstyle': '-', 'lw': county_plot.loc[kk, 'weight'],
                            'color': 'blue', 'alpha': 0.8, 'connectionstyle': "arc3,rad=0.2"}, va='center')
ax.axis('off')
plt.tight_layout()
plt.savefig(r'D:\OD_Social_Connect\Results\Weight_Distance.png', dpi=1000)
plt.close()

# # Method 2: Pairwise OD matrix
'''
week_visit = pd.concat(map(pd.read_csv, glob.glob(os.path.join(
    r'F:\\SafeGraph\\Monthly Places Patterns (aka Patterns) Dec 2020 - Present\\release-2021-07\\patterns\\2022\\04\\05\\20',
    "*.gz"))))
week_visit = week_visit[['poi_cbg', 'visitor_home_cbgs']]
week_visit = week_visit.dropna(subset=['poi_cbg']).reset_index(drop=True)
week_visit = week_visit[~week_visit['poi_cbg'].astype(str).str.contains('[A-Za-z]')].reset_index(drop=True)
week_visit['poi_cbg'] = week_visit['poi_cbg'].astype('int64').astype(str).apply(lambda x: x.zfill(12))
# Iterate visit flows
flows_unit = []
for i, row in enumerate(week_visit.itertuples()):
    if row.visitor_home_cbgs == "{}":
        continue
    else:
        origin = eval(row.visitor_home_cbgs)
        destination = row.poi_cbg
        for key, value in origin.items(): flows_unit.append([str(key).zfill(12), str(destination).zfill(12), value])
cbg_visits_flow_all = pd.DataFrame(flows_unit, columns=["cbg_o", "cbg_d", "visitor_flows"])
cbg_visits_flow_all['O_CTFIPS'] = cbg_visits_flow_all['cbg_o'].str[0:5]
cbg_visits_flow_all['D_CTFIPS'] = cbg_visits_flow_all['cbg_d'].str[0:5]
# CT flow
day_od = cbg_visits_flow_all.groupby(['O_CTFIPS', 'D_CTFIPS'])['visitor_flows'].sum().reset_index()
day_od.to_pickle(r'C:\\Users\\huson\\PycharmProjects\\OD_Social_Connect\\data\\OD_Flow.pkl')

week_visit = pd.concat(map(pd.read_csv, glob.glob(os.path.join(
    r'F:\\SafeGraph\\Monthly Places Patterns (aka Patterns) Dec 2020 - Present\\release-2021-07\\patterns_backfill\\2021\\07\\15\\16\\2020\\04',
    "*.gz"))))
week_visit = week_visit[['poi_cbg', 'visitor_home_cbgs']]
week_visit = week_visit.dropna(subset=['poi_cbg']).reset_index(drop=True)
week_visit = week_visit[~week_visit['poi_cbg'].astype(str).str.contains('[A-Za-z]')].reset_index(drop=True)
week_visit['poi_cbg'] = week_visit['poi_cbg'].astype('int64').astype(str).apply(lambda x: x.zfill(12))
# Iterate visit flows
flows_unit = []
for i, row in enumerate(week_visit.itertuples()):
    if row.visitor_home_cbgs == "{}":
        continue
    else:
        origin = eval(row.visitor_home_cbgs)
        destination = row.poi_cbg
        for key, value in origin.items(): flows_unit.append([str(key).zfill(12), str(destination).zfill(12), value])
cbg_visits_flow_all = pd.DataFrame(flows_unit, columns=["cbg_o", "cbg_d", "visitor_flows"])
cbg_visits_flow_all['O_CTFIPS'] = cbg_visits_flow_all['cbg_o'].str[0:5]
cbg_visits_flow_all['D_CTFIPS'] = cbg_visits_flow_all['cbg_d'].str[0:5]
# CT flow
day_od = cbg_visits_flow_all.groupby(['O_CTFIPS', 'D_CTFIPS'])['visitor_flows'].sum().reset_index()
day_od.to_pickle(r'C:\\Users\\huson\\PycharmProjects\\OD_Social_Connect\\data\\OD_Flow_202004.pkl')
'''

# # Weight normal by county
day_od = pd.read_pickle(r'C:\Users\huson\PycharmProjects\OD_Social_Connect\data\OD_Flow.pkl')
inflow_agg = day_od.groupby(['D_CTFIPS']).sum()['visitor_flows'].reset_index()
inflow_agg.columns = ['D_CTFIPS', 'inflow_agg']
day_od = day_od.merge(inflow_agg, on='D_CTFIPS')
day_od['inflow_pct'] = day_od['visitor_flows'] / day_od['inflow_agg']

# Threshold
day_od['weight'] = 0
t_h = 0.1
print('No of edges: %s' % (len(day_od.loc[day_od['inflow_pct'] >= t_h, 'weight'])))
day_od.loc[day_od['inflow_pct'] >= t_h, 'weight'] = day_od.loc[day_od['inflow_pct'] >= t_h, 'inflow_pct']
day_od = day_od[(day_od['O_CTFIPS'].isin(set(ct_visit["CTFIPS"]))) & (day_od['D_CTFIPS'].isin(set(ct_visit["CTFIPS"])))]
day_od.loc[day_od['O_CTFIPS'] == day_od['D_CTFIPS'], 'weight'] = 1

# To array and output
county_dist_wide = day_od.pivot(index='O_CTFIPS', columns='D_CTFIPS', values='weight')
county_dist_wide = county_dist_wide.fillna(0)
county_dist_wide = county_dist_wide.astype(np.float32)
A_c = county_dist_wide.to_numpy()
np.save(r'C:\Users\huson\PycharmProjects\OD_Social_Connect\data\us_adj_mat_od_pre.npy', A_c)

# Plot
county_plot = day_od[(day_od['weight'] != 0)].reset_index(drop=True)
county_ll.columns = ['O_CTFIPS', 'O_Lng', 'O_Lat']
county_plot = county_plot.merge(county_ll, on='O_CTFIPS')
county_ll.columns = ['D_CTFIPS', 'D_Lng', 'D_Lat']
county_plot = county_plot.merge(county_ll, on='D_CTFIPS')
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(9, 6))
poly.geometry.boundary.plot(color=None, edgecolor='gray', linewidth=0.2, ax=ax)
for kk in range(0, len(county_plot)):
    ax.annotate('', xy=(county_plot.loc[kk, 'O_Lng'], county_plot.loc[kk, 'O_Lat']),
                xytext=(county_plot.loc[kk, 'D_Lng'], county_plot.loc[kk, 'D_Lat']),
                arrowprops={'arrowstyle': '-', 'lw': county_plot.loc[kk, 'weight'] * 10,
                            'color': 'blue', 'alpha': 0.8, 'connectionstyle': "arc3,rad=0.2"}, va='center')
ax.axis('off')
plt.tight_layout()
plt.savefig(r'D:\OD_Social_Connect\Results\Weight_OD_Pct.png', dpi=1000)
plt.close()

# # Method 3: FB friendship
FB_F = pd.read_csv(r'D:\OD_Social_Connect\FB_Social\county_county.tsv', sep='\t')
FB_F['user_loc'] = FB_F['user_loc'].astype('int64').astype(str).apply(lambda x: x.zfill(5))
FB_F['fr_loc'] = FB_F['fr_loc'].astype('int64').astype(str).apply(lambda x: x.zfill(5))
FB_F.columns = ['O_CTFIPS', 'D_CTFIPS', 'SCI']

inflow_agg = FB_F.groupby(['D_CTFIPS']).sum()['SCI'].reset_index()
inflow_agg.columns = ['D_CTFIPS', 'inflow_agg']
FB_F = FB_F.merge(inflow_agg, on='D_CTFIPS')
FB_F['inflow_pct'] = FB_F['SCI'] / FB_F['inflow_agg']

# Threshold
FB_F['weight'] = 0
t_h = 0.1
print('No of edges: %s' % (len(FB_F.loc[FB_F['inflow_pct'] >= t_h, 'weight'])))
FB_F.loc[FB_F['inflow_pct'] >= t_h, 'weight'] = FB_F.loc[FB_F['inflow_pct'] >= t_h, 'inflow_pct']
FB_F = FB_F[(FB_F['O_CTFIPS'].isin(set(ct_visit["CTFIPS"]))) & (FB_F['D_CTFIPS'].isin(set(ct_visit["CTFIPS"])))]
FB_F.loc[FB_F['O_CTFIPS'] == FB_F['D_CTFIPS'], 'weight'] = 1

# To array and output
county_dist_wide = FB_F.pivot(index='O_CTFIPS', columns='D_CTFIPS', values='weight')
county_dist_wide = county_dist_wide.fillna(0)
county_dist_wide = county_dist_wide.astype(np.float32)
A_c = county_dist_wide.to_numpy()
np.save(r'C:\Users\huson\PycharmProjects\OD_Social_Connect\data\us_adj_mat_fb_pre.npy', A_c)

# Plot
county_plot = FB_F[(FB_F['weight'] != 0)].reset_index(drop=True)
county_ll.columns = ['O_CTFIPS', 'O_Lng', 'O_Lat']
county_plot = county_plot.merge(county_ll, on='O_CTFIPS')
county_ll.columns = ['D_CTFIPS', 'D_Lng', 'D_Lat']
county_plot = county_plot.merge(county_ll, on='D_CTFIPS')
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(9, 6))
poly.geometry.boundary.plot(color=None, edgecolor='gray', linewidth=0.2, ax=ax)
for kk in range(0, len(county_plot)):
    ax.annotate('', xy=(county_plot.loc[kk, 'O_Lng'], county_plot.loc[kk, 'O_Lat']),
                xytext=(county_plot.loc[kk, 'D_Lng'], county_plot.loc[kk, 'D_Lat']),
                arrowprops={'arrowstyle': '-', 'lw': county_plot.loc[kk, 'weight'] * 10,
                            'color': 'blue', 'alpha': 0.8, 'connectionstyle': "arc3,rad=0.2"}, va='center')
ax.axis('off')
plt.tight_layout()
plt.savefig(r'D:\OD_Social_Connect\Results\Weight_FB_Pct.png', dpi=1000)
plt.close()

# # Method 4: Gravity model
county_re = pd.DataFrame(np.repeat(county_ll.to_numpy(), len(county_ll), axis=0), columns=county_ll.columns)
county_re.columns = ['O_CTFIPS', 'O_Lng', 'O_Lat']
county_re_1 = county_ll.iloc[np.tile(np.arange(len(county_ll)), len(county_ll))].reset_index(drop=True)
county_re_1.columns = ['D_CTFIPS', 'D_Lng', 'D_Lat']
county_re = pd.concat([county_re, county_re_1], axis=1)
del county_re_1
county_re[['O_Lng', 'O_Lat', 'D_Lng', 'D_Lat']] = county_re[['O_Lng', 'O_Lat', 'D_Lng', 'D_Lat']].astype(float)
county_re['Dist'] = haversine_array(county_re['O_Lat'], county_re['O_Lng'], county_re['D_Lat'], county_re['D_Lng']) + 1
county_re['Dist_exp'] = np.exp(-county_re['Dist'] ** 2 / np.mean(county_re['Dist']) ** 2)

# population
county_pp = ct_visit.drop_duplicates(subset=['CTFIPS']).reset_index(drop=True)[['CTFIPS', 'Total_Population']]
county_pp.columns = ['O_CTFIPS', 'O_POP']
county_re = county_re.merge(county_pp, on='O_CTFIPS')
county_pp.columns = ['D_CTFIPS', 'D_POP']
county_re = county_re.merge(county_pp, on='D_CTFIPS')
county_re['Att'] = (county_re['O_POP'] * county_re['D_POP']) / ((county_re['Dist']) ** 1.1)

county_re['max_att'] = county_re.groupby('O_CTFIPS').Att.transform(np.max)
county_re['att_pct'] = county_re['Att'] / county_re['max_att']

# Threshold
county_re['weight_att'] = 0
t_h = 0.1
print('No of edges: %s' % (len(county_re.loc[county_re['att_pct'] >= t_h, 'weight_att'])))
county_re.loc[county_re['att_pct'] >= t_h, 'weight_att'] = county_re.loc[county_re['att_pct'] >= t_h, 'att_pct']
county_re.loc[county_re['O_CTFIPS'] == county_re['D_CTFIPS'], 'weight_att'] = 1

# To array and output
county_dist_wide = county_re.pivot(index='O_CTFIPS', columns='D_CTFIPS', values='weight_att')
county_dist_wide = county_dist_wide.fillna(0)
county_dist_wide = county_dist_wide.astype(np.float32)
A_c = county_dist_wide.to_numpy()
np.save(r'C:\Users\huson\PycharmProjects\OD_Social_Connect\data\us_adj_mat_gravity_pre.npy', A_c)

# Plot graph
county_plot = county_re[
    (county_re['weight_att'] != 0) & (county_re['weight_att'] < county_re['weight_att'].max())].reset_index(drop=True)
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(9, 6))
poly.geometry.boundary.plot(color=None, edgecolor='gray', linewidth=0.2, ax=ax)
for kk in range(0, len(county_plot)):
    ax.annotate('', xy=(county_plot.loc[kk, 'O_Lng'], county_plot.loc[kk, 'O_Lat']),
                xytext=(county_plot.loc[kk, 'D_Lng'], county_plot.loc[kk, 'D_Lat']),
                arrowprops={'arrowstyle': '-', 'lw': county_plot.loc[kk, 'weight_att'],
                            'color': 'blue', 'alpha': 0.8, 'connectionstyle': "arc3,rad=0.2"}, va='center')
ax.axis('off')
plt.tight_layout()
plt.savefig(r'D:\OD_Social_Connect\Results\Weight_Gravity.png', dpi=1000)
plt.close()
