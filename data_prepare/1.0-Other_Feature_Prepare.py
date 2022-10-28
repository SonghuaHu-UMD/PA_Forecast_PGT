import datetime
import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
import geopandas as gpd
import seaborn as sns
import glob
import os

# Parameter setting
plt.rcParams.update(
    {'font.size': 13, 'font.family': "serif", 'mathtext.fontset': 'dejavuserif', 'xtick.direction': 'in',
     'xtick.major.size': 0.5, 'grid.linestyle': "--", 'axes.grid': True, "grid.alpha": 1, "grid.color": "#cccccc",
     'xtick.minor.size': 1.5, 'xtick.minor.width': 0.5, 'xtick.minor.visible': True, 'xtick.top': True,
     'ytick.direction': 'in', 'ytick.major.size': 0.5, 'ytick.minor.size': 1.5, 'ytick.minor.width': 0.5,
     'ytick.minor.visible': True, 'ytick.right': True, 'axes.linewidth': 0.5, 'grid.linewidth': 0.5,
     'lines.linewidth': 1.5, 'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05})
t_start = datetime.datetime(2018, 5, 1)
t_end = datetime.datetime(2022, 1, 31)
pd.options.mode.chained_assignment = None
un_st = ['00', '02', '15', '60', '66', '69', '72', '78']

######### All CT visits #########
# Read CT visits
day_visit_ct = pd.concat(
    map(pd.read_pickle, glob.glob(os.path.join(r'D:\\OD_Social_Connect\\Neighbor_stops_ct\\', "*.pkl"))))
day_visit_ct['Date'] = pd.to_datetime(day_visit_ct['Date'])
day_visit_ct = day_visit_ct[(day_visit_ct['Date'] >= t_start) & (day_visit_ct['Date'] <= t_end)]
# day_visit_ct.groupby('Date')['Daily_Flow'].sum().plot()

# Drop those small CT
Total_CT = day_visit_ct.groupby(['CTFIPS'])['Daily_Flow'].sum().reset_index()
need_ct = Total_CT.loc[Total_CT['Daily_Flow'] >= (t_end - t_start).days * 3, 'CTFIPS']
day_visit_ct = day_visit_ct.merge(need_ct, on='CTFIPS').reset_index(drop=True)

# Drop some states
day_visit_ct['STFIPS'] = day_visit_ct['CTFIPS'].str[0:2]
day_visit_ct = day_visit_ct[~day_visit_ct['STFIPS'].isin(un_st)].reset_index(drop=True)

# Full Time range
timerange = pd.DataFrame(pd.date_range(start=t_start, end=t_end, freq='D'))
timerange[0] = timerange[0].astype(str)
full_time_range = pd.DataFrame(
    {'CTFIPS': np.repeat(list(need_ct), len(timerange)), 'Date': list(timerange[0]) * len(need_ct)})
full_time_range['Date'] = pd.to_datetime(full_time_range['Date'])
ct_visit = day_visit_ct.merge(full_time_range, on=['CTFIPS', 'Date'], how='right')
ct_visit = ct_visit.fillna(0)
del full_time_range
print(len(ct_visit) / len(need_ct))

# Is holidays
holidays = USFederalHolidayCalendar().holidays(start=t_start, end=t_end).to_pydatetime()
ct_visit['Is_holiday'] = ct_visit['Date'].isin(holidays)
ct_visit['Is_holiday'] = ct_visit['Is_holiday'].astype(int)

# Time index
ct_visit['dayofweek'] = ct_visit['Date'].dt.dayofweek
ct_visit["month"] = ct_visit['Date'].dt.month
ct_visit["time_idx"] = (ct_visit["Date"] - ct_visit.Date.min()).dt.days
ct_visit['Isweekend'] = 0
ct_visit.loc[ct_visit['dayofweek'].isin(['5', '6']), 'Isweekend'] = 1
print(ct_visit.isnull().sum())

######### Weather #########
# https://www.ncei.noaa.gov/data/global-summary-of-the-day/archive/2020
weather_raw_2018 = pd.concat(map(pd.read_csv, glob.glob(os.path.join('D:\\Vaccination\\Weather\\2018\\', "*.csv"))))
weather_raw_2018.to_pickle('D:\\Vaccination\\Weather\\weather_raw_2018.pkl')
weather_raw_2019 = pd.concat(map(pd.read_csv, glob.glob(os.path.join('D:\\Vaccination\\Weather\\2019\\', "*.csv"))))
weather_raw_2019.to_pickle('D:\\Vaccination\\Weather\\weather_raw_2019.pkl')
weather_raw_2020 = pd.concat(map(pd.read_csv, glob.glob(os.path.join('D:\\Vaccination\\Weather\\2020\\', "*.csv"))))
weather_raw_2020.to_pickle('D:\\Vaccination\\Weather\\weather_raw_2020.pkl')
weather_raw_2021 = pd.concat(map(pd.read_csv, glob.glob(os.path.join('D:\\Vaccination\\Weather\\2021\\', "*.csv"))))
weather_raw_2021.to_pickle('D:\\Vaccination\\Weather\\weather_raw_2021.pkl')
weather_raw_2022 = pd.concat(map(pd.read_csv, glob.glob(os.path.join('D:\\Vaccination\\Weather\\2022\\', "*.csv"))))
weather_raw_2022.to_pickle('D:\\Vaccination\\Weather\\weather_raw_2022.pkl')
weather_raw = pd.concat(
    [weather_raw_2018, weather_raw_2019, weather_raw_2020, weather_raw_2021, weather_raw_2022]).reset_index(drop=True)
del weather_raw_2018, weather_raw_2019, weather_raw_2020, weather_raw_2021, weather_raw_2022

# Station in US
g_stat = weather_raw[['STATION', 'LATITUDE', 'LONGITUDE']]
g_stat = g_stat.drop_duplicates(subset=['STATION']).dropna()
g_stat_s = gpd.GeoDataFrame(g_stat, geometry=gpd.points_from_xy(g_stat['LONGITUDE'], g_stat['LATITUDE']))
ghcnd_station_s = g_stat_s.set_crs('EPSG:4326')
poly_raw = gpd.GeoDataFrame.from_file(
    r'F:\\SafeGraph\\Open Census Data\\Census Website\\2019\\nhgis0011_shape\\US_county_2019.shp')
poly_raw = poly_raw.to_crs('EPSG:4326')
poly_raw['CTFIPS'] = poly_raw['GISJOIN'].str[1:3] + poly_raw['GISJOIN'].str[4:7]
SInUS = gpd.sjoin(ghcnd_station_s, poly_raw, how='inner', op='within').reset_index(drop=True)
SInUS = SInUS[['STATION', 'LATITUDE', 'LONGITUDE', 'CTFIPS']]

# Weather info in US
weather_raw = weather_raw[weather_raw['STATION'].isin(SInUS['STATION'])].reset_index(drop=True)
weather_raw.loc[weather_raw['TEMP'] == 9999, 'TEMP'] = np.nan
weather_raw.loc[weather_raw['PRCP'] == 99.99, 'PRCP'] = 0
weather_raw['Date'] = pd.to_datetime(weather_raw['DATE'])
weather_raw = weather_raw.merge(SInUS[['STATION', 'CTFIPS']], on='STATION')

# Daily weather
weather_ct = weather_raw.groupby(['CTFIPS', 'Date'])[['TEMP', 'PRCP']].mean().reset_index()
# weather_raw_d.groupby(['DATE'])[['PRCP']].mean().plot()
weather_ct['STFIPS'] = weather_ct['CTFIPS'].str[0:2]
weather_ct = weather_ct[~weather_ct['STFIPS'].isin(un_st)].reset_index(drop=True)
weather_st = weather_ct.groupby(['STFIPS', 'Date'])[['TEMP', 'PRCP']].mean().reset_index()
weather_st.columns = ['STFIPS', 'Date', 'ST_TEMP', 'ST_PRCP']

# Merge with weather
ct_visit = ct_visit.merge(weather_ct[['CTFIPS', 'Date', 'TEMP', 'PRCP']], on=['CTFIPS', 'Date'], how='left')
ct_visit['STFIPS'] = ct_visit['CTFIPS'].str[0:2]
ct_visit = ct_visit.merge(weather_st, on=['STFIPS', 'Date'])
ct_visit.loc[ct_visit['TEMP'].isnull(), 'TEMP'] = ct_visit.loc[ct_visit['TEMP'].isnull(), 'ST_TEMP']
ct_visit.loc[ct_visit['PRCP'].isnull(), 'PRCP'] = ct_visit.loc[ct_visit['PRCP'].isnull(), 'ST_PRCP']
ct_visit.isnull().sum()

######### CT Features #########
# Read CT features
CT_Features = pd.read_csv(r'D:\COVID19-Socio\Data\County_COVID_19.csv', index_col=0)
CT_Features['CTFIPS'] = CT_Features['CTFIPS'].astype(str).apply(lambda x: x.zfill(5))
CT_Features['Agriculture_Mining_R'] = \
    CT_Features['Agriculture_R'] + CT_Features['Mining_R'] + CT_Features['Construction_R']
CT_Features['Transportation_Utilities_R'] = \
    CT_Features['Transportation_R'] + CT_Features['Utilities_R']
CT_Features['Retail_Wholesale_R'] = CT_Features['Retail_R'] + CT_Features['Wholesale_R']
CT_Features['Administrative_Management_R'] = CT_Features['Management_R'] + CT_Features['Administrative_R']
CT_Features['Accommodation_food_arts_R'] = CT_Features['Accommodation_food_R'] + CT_Features['Arts_R']
CT_Features['Finance_R'] = CT_Features['Finance_R'] + CT_Features['Real_estate_R']
CT_Features['Indian_Others_R'] = 100 - (CT_Features['Asian_R'] + CT_Features['White_R'] + CT_Features['Black_R'])

# Update voting data
Election = pd.read_csv(r'D:\Vaccination\countypres_2000-2020.csv')
Election = Election[(Election['year'] == 2020)]  # & (Election['mode'] == 'TOTAL')
Election = Election[['county_fips', 'party', 'candidatevotes', 'totalvotes']]
Election = Election.dropna(subset=['county_fips'])
Election['county_fips'] = Election['county_fips'].astype(int).astype(str).apply(lambda x: x.zfill(5))
Election['candidatevotes'] = Election['candidatevotes'].fillna(0)

# Total vote
Election_total = Election.drop_duplicates(subset=['county_fips'])
Election_total = Election_total[['county_fips', 'totalvotes']]
Election_total.columns = ['CTFIPS', 'Total_votes']
Election_total_s = Election.groupby(['county_fips']).sum()['candidatevotes'].reset_index()
Election_total_s.columns = ['CTFIPS', 'Total_votes_s']
Election_total = Election_total.merge(Election_total_s, on='CTFIPS')
Election_total = Election_total[['CTFIPS', 'Total_votes_s']]

Election_democrat = Election[Election['party'] == 'DEMOCRAT']
Election_democrat = Election_democrat[['county_fips', 'candidatevotes']]
Election_democrat = Election_democrat.groupby(['county_fips']).sum().reset_index()
Election_democrat.columns = ['CTFIPS', 'Democrat']

Election_republican = Election[Election['party'] == 'REPUBLICAN']
Election_republican = Election_republican[['county_fips', 'candidatevotes']]
Election_republican = Election_republican.groupby(['county_fips']).sum().reset_index()
Election_republican.columns = ['CTFIPS', 'Republican']

Election = Election_democrat.merge(Election_republican, on='CTFIPS')
Election = Election.merge(Election_total, on='CTFIPS')

Election = Election.dropna(subset=['CTFIPS'])
Election['CTFIPS'] = Election['CTFIPS'].astype(int).astype(str).apply(lambda x: x.zfill(5))
Election['Democrat_R'] = Election['Democrat'] / Election['Total_votes_s'] * 100
Election['Republican_R'] = Election['Republican'] / Election['Total_votes_s'] * 100
# sum(Election['Democrat']) / sum(Election['Total_votes_s'])

# Add some variable from SVI
SVI_2018 = pd.read_csv(r'D:\Vaccination\SVI\SVI2018_US_COUNTY.csv')
SVI_2018['CTFIPS'] = SVI_2018['FIPS'].astype(int).astype(str).apply(lambda x: x.zfill(5))
SVI_2018 = SVI_2018[
    ['CTFIPS', 'EP_NOHSDP', 'EP_DISABL', 'EP_SNGPNT', 'EP_LIMENG', 'EP_MUNIT', 'EP_MOBILE', 'EP_CROWD', 'EP_NOVEH',
     'EP_GROUPQ']]
SVI_2018.columns = ['CTFIPS', 'No_Highschool_R', 'Disability_R', 'Single_parent_R', 'Less_English_R', 'Munit_house_R',
                    'Mobile_Home_R', 'Crowd_Home_R', 'No_vehicle_R', 'Group_Quarters_R']

# Merge with county features
ct_visit = ct_visit.merge(
    CT_Features[
        ['CTFIPS', 'Total_Population', 'Median_income', 'Rent_to_Income', 'GINI', 'Agriculture_Mining_R', 'Finance_R',
         'Manufacturing_R', 'Retail_Wholesale_R', 'Transportation_Utilities_R', 'Information_R', 'Scientific_R',
         'Administrative_Management_R', 'Educational_R', 'Health_care_R', 'ALAND', 'Lng', 'Lat', 'Is_Central', 'STUSPS',
         'Accommodation_food_arts_R', 'Urbanized_Areas_Population_R', 'Urban_Clusters_Population_R', 'Worked_at_home_R',
         'Rural_Population_R', 'No_Insurance_R', 'Household_Below_Poverty_R', 'White_Non_Hispanic_R', 'Black_R',
         'Asian_R', 'HISPANIC_LATINO_R', 'Indian_Others_R', 'Under_18_R', 'Bt_18_44_R', 'Bt_45_64_R', 'Over_65_R',
         'Male_R', 'Population_Density', 'Education_Degree_R', 'Unemployed_R']], on='CTFIPS')
ct_visit = ct_visit.merge(SVI_2018, on='CTFIPS')
ct_visit = ct_visit.merge(Election[['CTFIPS', 'Democrat_R', 'Republican_R']], on='CTFIPS', how='left')
ct_visit['Democrat_R'] = ct_visit['Democrat_R'].fillna(ct_visit['Democrat_R'].mean())
ct_visit['Republican_R'] = ct_visit['Republican_R'].fillna(ct_visit['Republican_R'].mean())
print(ct_visit.isnull().sum().sum())
print(len(set(ct_visit['CTFIPS'])))
print(len(ct_visit) / len(set(ct_visit['CTFIPS'])))
ct_visit = ct_visit.dropna().reset_index(drop=True)
ct_visit.to_pickle(r'D:\OD_Social_Connect\Data\ct_visit.pkl')

ct_visit = pd.read_pickle(r'D:\OD_Social_Connect\Data\ct_visit.pkl')
