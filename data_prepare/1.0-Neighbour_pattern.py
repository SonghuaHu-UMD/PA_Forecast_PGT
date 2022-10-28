import pandas as pd
import os
import datetime
import glob

pd.options.mode.chained_assignment = None

# Neighbourhood pattern
range_year = [d.strftime('%Y') for d in
              pd.date_range(datetime.datetime(2018, 1, 1), datetime.datetime(2022, 2, 1), freq='M')]
range_month = [int(d.strftime('%m')) for d in
               pd.date_range(datetime.datetime(2018, 1, 1), datetime.datetime(2022, 2, 1), freq='M')]
for jj in range(0, len(range_year)):  # 18: 2019/04/22; 25: 2019/06/10 0, len(range_year)
    start = datetime.datetime.now()
    print(str(jj) + '_' + str(range_year[jj]) + '\\' + str(range_month[jj]) + '\\')
    # change to the deepest subdir
    for dirpaths, dirnames, filenames in os.walk(
            "F:\\SafeGraph\\Neighbourhood Patterns\\neighborhood-patterns\\2022\\02\\09\\release-2021-07-01\\neighborhood_patterns\\y="
            + str(range_year[jj]) + '\\m=' + str(range_month[jj])):
        if not dirnames: os.chdir(dirpaths)
    monthly_visit = pd.concat(map(pd.read_csv, glob.glob(os.path.join('', "*.gz"))))
    day_visit = pd.DataFrame(monthly_visit['stops_by_day'].str[1:-1].str.split(',').tolist()).astype(int)
    date_range = [d.strftime('%Y-%m-%d')
                  for d in pd.date_range(monthly_visit.loc[0, 'date_range_start'].split('T')[0],
                                         monthly_visit.loc[0, 'date_range_end'].split('T')[0], freq='d')][0: -1]
    day_visit.columns = date_range
    day_visit['CBFIPS'] = monthly_visit['area']
    # day_visit_st = day_visit.groupby(['CBFIPS']).sum().unstack().reset_index()
    day_visit_st = pd.melt(day_visit, id_vars=['CBFIPS'], value_vars=date_range)
    day_visit_st.columns = ['CBFIPS', 'Date', 'Daily_Flow']
    day_visit_st['CBFIPS'] = day_visit_st['CBFIPS'].astype('int64').astype(str).apply(lambda x: x.zfill(12))
    # Ouput
    day_visit_st.to_pickle(
        r'D:\\OD_Social_Connect\\Neighbor_stops_cbg\\Visit_CBG_%s_%s.pkl' % (range_year[jj], range_month[jj]))
    print(datetime.datetime.now() - start)

    # All in county level
    day_visit_st['CTFIPS'] = day_visit_st['CBFIPS'].str[0:5]
    day_visit_st = day_visit_st.drop('CBFIPS', axis=1)
    day_visit_st = day_visit_st.groupby(['CTFIPS', 'Date'])['Daily_Flow'].sum().reset_index()
    day_visit_st.to_pickle(
        r'D:\\OD_Social_Connect\\Neighbor_stops_ct\\Visit_CT_%s_%s.pkl' % (range_year[jj], range_month[jj]))
