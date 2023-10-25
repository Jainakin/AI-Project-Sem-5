import ImportAndClean

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

# plt.style.use('seaborn')

crimes = pd.read_csv(r"C:\Users\suran\Desktop\School\1 UNIVERSITY\BENNETT\CSET204 Prob and Stats\Hackathon Predictive policing\Hackathon Predictive policing\Chicago_Crimes_2012_to_2017.csv\Chicago_Crimes_2012_to_2017.csv")
# trends visualised
def visualize():
    data_feed = ['IUCR  == "0820" or IUCR == "0810"', 'IUCR  == "0460" or IUCR == "0486"', 'IUCR  == "1320"']
    crime_type = ["Theft", "Battery", "Criminal Damage"]
    crimeImported = ImportAndClean.cleaning()
    for i in range(0, 3):
        IUCR_data = crimeImported.query(data_feed[i])
        plt.figure(figsize=(11,5))
        IUCR_data.resample('M').size().plot(legend=False)
        plt.title('Number of {} per month (2012 - 2017)'.format(crime_type[i]))
        plt.xlabel('Months')
        plt.ylabel('Number of {}'.format(crime_type[i]))
        plt.show()
            
#heat map visuals
crimes = ImportAndClean.cleaning()
days = ['Monday','Tuesday','Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
crimes.groupby([crimes.index.dayofweek]).size().plot(kind='barh')
plt.ylabel('Days of the week')
plt.yticks(np.arange(7), days)
plt.xlabel('Number of crimes')
plt.title('Number of crimes by day of the week')
plt.show()

crimes.groupby([crimes.index.month]).size().plot(kind='barh')
plt.ylabel('Months of the year')
plt.xlabel('Number of crimes')
plt.title('Number of crimes by month of the year')
plt.show()

plt.figure(figsize=(8,10))
crimes.groupby([crimes['Primary Type']]).size().sort_values(ascending=True).plot(kind='barh')
plt.title('Number of crimes by type')
plt.ylabel('Crime Type')
plt.xlabel('Number of crimes')
plt.show()
hour_by_location = crimes.pivot_table(values='ID', index='Location Description', columns=crimes.index.hour, aggfunc=np.size).fillna(0)
hour_by_type     = crimes.pivot_table(values='ID', index='Primary Type', columns=crimes.index.hour, aggfunc=np.size).fillna(0)
# hour_by_week     = crimes.pivot_table(values='ID', index=crimes.index.hour, columns=crimes.index.weekday, aggfunc=np.size).fillna(0)
# hour_by_week     = hour_by_week[days].T # sort cols by order of days
dayofweek_by_location = crimes.pivot_table(values='ID', index='Location Description', columns=crimes.index.dayofweek, aggfunc=np.size).fillna(0)
dayofweek_by_type = crimes.pivot_table(values='ID', index='Primary Type', columns=crimes.index.dayofweek, aggfunc=np.size).fillna(0)
location_by_type  = crimes.pivot_table(values='ID', index='Location Description', columns='Primary Type', aggfunc=np.size).fillna(0)

#clustering method to process heatmap
from sklearn.cluster import AgglomerativeClustering as AC


def scale_df(df,axis=0):
    '''
    A utility function to scale numerical values (z-scale) to have a mean of zero
    and a unit variance.
    '''
    return (df - df.mean(axis=axis)) / df.std(axis=axis)

def plot_hmap(df, ix=None, cmap='bwr'):
    '''
    A function to plot heatmaps that show temporal patterns
    '''
    if ix is None:
        ix = np.arange(df.shape[0])
    plt.imshow(df.iloc[ix,:], cmap=cmap)
    plt.colorbar(fraction=0.03)
    plt.yticks(np.arange(df.shape[0]), df.index[ix])
    plt.xticks(np.arange(df.shape[1]))
    plt.grid(False)
    plt.show()
    
def scale_and_plot(df, ix = None):
    '''
    A wrapper function to calculate the scaled values within each row of df and plot_hmap
    '''
    df_marginal_scaled = scale_df(df.T).T
    if ix is None:
        ix = AC(4).fit(df_marginal_scaled).labels_.argsort() # a trick to make better heatmaps
    cap = np.min([np.max(df_marginal_scaled.as_matrix()), np.abs(np.min(df_marginal_scaled.as_matrix()))])
    df_marginal_scaled = np.clip(df_marginal_scaled, -1*cap, cap)
    plot_hmap(df_marginal_scaled, ix=ix)
    
def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

crimes_new = crimes[(crimes['Primary Type'] == 'THEFT') | (crimes['Primary Type'] == 'BATTERY') | (crimes['Primary Type'] == 'CRIMINAL DAMAGE')]
crimesPls = crimes_new[['Longitude','Latitude']]
ax = sns.lmplot(data = crimesPls, x = 'Longitude', y = 'Latitude',fit_reg=False, height=4, scatter_kws={'alpha':.1})
ax = sns.kdeplot(data = crimesPls,
                # cmap = "Reds",
                # shade = True,
                # hue_norm=(0,255),
                bw_method =.005,
                #n_levels=10,
                #  cbar=True, 
                color = "red",
                fill = False,
                thresh = False)
ax.set_xlim(-87.9,-87.5)
ax.set_ylim(41.60,42.05)
ax.set_axis_off()
plt.show()      