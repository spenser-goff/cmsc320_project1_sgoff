import requests, pandas as pd, numpy as np, time, datetime, matplotlib.pyplot as plt, matplotlib.dates
from bs4 import BeautifulSoup
"""-----------------------------------------------------------------------"""      
#Part 1: Data Scraping and preparation
"""-----------------------------------------------------------------------"""      
#Step 1: Scrape your competitor’s data
r = requests.get('https://cmsc320.github.io/files/top-50-solar-flares.html')
root = BeautifulSoup(r.content)
root.prettify()
#table = root.find("div", id="SWL_Page").find("table").find("tbody")
tables = pd.read_html('https://cmsc320.github.io/files/top-50-solar-flares.html')
df_swl = tables[0]
df_swl = df_swl.rename(columns={"Unnamed: 0":"rank", "Unnamed: 1":"x_classification",
                        "Unnamed: 2":"date", "Region":"region", "Start":"start_time", 
                       "Maximum":"maximum_time", "End":"end_time", "Unnamed: 7":"movie"})
"""-----------------------------------------------------------------------"""      
#Step 2: Tidy the top 50 solar flare data
df_swl = df_swl.drop(columns=['movie'])
for index, row in df_swl.iterrows():
    startdate_str = str(row['date']) + " " + str(row['start_time']) + ":00"
    maxdate_str = str(row['date']) + " " + str(row['maximum_time']) + ":00"
    enddate_str = str(row['date']) + " " + str(row['end_time']) + ":00"
    startdate_str = startdate_str.replace('/', '-')
    maxdate_str = maxdate_str.replace('/', '-')
    enddate_str = enddate_str.replace('/', '-')
    startdate_obj = datetime.datetime.strptime(startdate_str, '%Y-%m-%d %H:%M:%S')
    maxdate_obj = datetime.datetime.strptime(maxdate_str, '%Y-%m-%d %H:%M:%S')
    enddate_obj = datetime.datetime.strptime(enddate_str, '%Y-%m-%d %H:%M:%S')
    df_swl.at[index, 'date'] = startdate_obj
    df_swl.at[index, 'start_time'] = startdate_obj  
    df_swl.at[index, 'maximum_time'] = maxdate_obj
    df_swl.at[index, 'end_time'] = enddate_obj
    if df_swl.at[index, 'region'] == '-':
        df_swl.replace(to_replace = 'region', value = np.nan)

"""-----------------------------------------------------------------------"""      
#Step 3
r = requests.get('http://www.hcbravo.org/IntroDataSci/misc/waves_type2.html')
root = BeautifulSoup(r.content)
rawtext = root.find("pre").get_text()
rawtext = rawtext.split('\n')
table = []


for item in rawtext[12:-3]:
    item = item.split()
   # print(item)
    table.append(item)
df_nasa =  pd.DataFrame(table)
df_nasa = df_nasa.iloc[: , : -9]
df_nasa = df_nasa.set_axis(['start_date', 'start_time', 'end_date', 'end_time', 'start_freq', 'end_freq', 'Loc', 'NOAA', 'Imp', 'cme_date', 'cme_time', 'CPA', 'cme_width', 'cme_speed', 'plots'], axis=1, inplace=False)
"""-----------------------------------------------------------------------"""
#Step 4: Tidy the NASA table
df_nasa['start_freq'] = df_nasa.start_freq.apply(lambda x: x if x != "????" else np.nan)
df_nasa['end_freq'] = df_nasa.end_freq.apply(lambda x: x if x != "????" else np.nan)
df_nasa['NOAA'] = df_nasa.NOAA.apply(lambda x: x if x != "-----" else np.nan)
df_nasa['Imp'] = df_nasa.Imp.apply(lambda x: x if x != "----" else np.nan)
df_nasa['is_halo'] = df_nasa.CPA.apply(lambda x: True if x == "Halo" else False)
df_nasa['CPA'] = df_nasa.CPA.apply(lambda x: x if x != "----" else np.nan)
df_nasa['CPA'] = df_nasa.CPA.apply(lambda x: x if x != "Halo" else np.nan)
df_nasa['width_lower_bound'] = df_nasa.cme_width.apply(lambda x: True if '>' in str(x) else False)
df_nasa['cme_width'] = df_nasa.cme_width.apply(lambda x: str(x)[1:] if '>' in str(x) else str(x))
df_nasa['cme_width'] = df_nasa.cme_width.apply(lambda x: x if x != "----" else np.nan)
df_nasa['cme_speed'] = df_nasa.cme_speed.apply(lambda x: x if x != "----" else np.nan)
df_nasa['Loc'] = df_nasa.Loc.apply(lambda x: np.nan if "back" in str(x).lower() else str(x))
df_nasa['cme_date'] = df_nasa.cme_date.apply(lambda x: np.nan if str(x) == "--/--" else str(x))
df_nasa['cme_time'] = df_nasa.cme_time.apply(lambda x: np.nan if str(x) == "--:--" else str(x))

startdate_list = []
for index, row in df_nasa.iterrows():
    startdate_str = str(row['start_date']) + " " + str(row['start_time']) + ":00"
    startdate_list.append(startdate_str)
    cmedate_str = startdate_str[0:5] + str(row['cme_date']) + " " + str(row['cme_time']) + ":00"
    enddate_str = startdate_str[0:5] + str(row['end_date']) + " " + str(row['end_time']) + ":00"
    startdate_str = startdate_str.replace('/', '-')
    cmedate_str = cmedate_str.replace('/', '-')
    enddate_str = enddate_str.replace('/', '-')
#    startdate_obj = datetime.datetime.strptime(startdate_str, '%Y-%m-%d %H:%M:%S')
#    cmedate_obj = datetime.datetime.strptime(cmedate_str, '%Y-%m-%d %H:%M:%S')
#    enddate_obj = datetime.datetime.strptime(enddate_str, '%Y-%m-%d %H:%M:%S')
    df_nasa.at[index, 'start_datetime'] = startdate_str
    df_nasa.at[index, 'end_datetime'] = cmedate_str 
    df_nasa.at[index, 'cme_datetime'] = enddate_str

df_nasa = df_nasa.drop(columns = ['start_time', 'start_date', 'end_date', 'end_time', 'cme_date', 'cme_time'])

"""-----------------------------------------------------------------------"""
#Part 2: Analysis
"""-----------------------------------------------------------------------"""
#Question 1: Replication
def classify(x):
    if x == x and x != "FILA" and str(x)[0] == 'X':
        return str(x)[1:]
    else:
        return "NaN"
    
df_tmp1 = df_nasa.copy(deep=True)   
df_tmp1['Imp_Classify'] = df_tmp1.Imp.apply(classify)
df_tmp1['Imp_Classify'] = df_tmp1['Imp_Classify'].astype(float)
df_tmp1.sort_values('Imp_Classify',inplace=True, ascending=False)
imps = df_tmp1.head(50)
print(imps)
print(df_swl.head(50))
"""-----------------------------------------------------------------------"""
#Question 2: Integration

nasa_entry = ["" for i in range(50)]
df_tmp2 = pd.DataFrame(nasa_entry, columns=['NASA Entry'])
df_tmp2 = df_tmp2.rename_axis("SWL Rank")
for index, row in df_tmp2.iterrows():
    df_tmp2.at[index, 'NASA Entry'] = imps.at[index, "start_datetime"]
df_tmp2

df_tmp3 = df_tmp1.copy(deep=True)   
tmp = np.arange(1,483)
tmp = tmp.tolist()
df_tmp3['SWL_Rank'] = tmp
df_tmp3.sort_values('SWL_Rank',inplace=True, ascending=True)
df_tmp3 = df_tmp3.drop(['Imp_Classify'], axis=1)
df_tmp3['SWL_Rank'] = df_tmp3.SWL_Rank.apply(lambda x: x if x <= 33 else np.nan)
df_tmp3.at[9, "SWL_Rank"] = np.nan
df_tmp3.at[144, "SWL_Rank"] = 32
df_tmp3.sort_values('SWL_Rank',inplace=True, ascending=True)
df_nasa = df_tmp3.copy(deep=True)
df_nasa.head(50)
"""-----------------------------------------------------------------------"""
#Question 3: Analysis
print(type(startdate_list))
print(len(startdate_list))
datetime_list = []
for item in startdate_list:
    item = item.replace('/', '-')
    print(item)
    item = datetime.datetime.strptime(item, '%Y-%m-%d %H:%M:%S')
    print(type(item))
    datetime_list.append(item)
X_plot = matplotlib.dates.date2num(datetime_list)
print(type(X_plot))
Y_plot = df_nasa['Loc'].values
Y_plot = Y_plot.astype(str)
print(type(Y_plot))
plt.plot_date(X_plot, Y_plot)
plt.show() 
