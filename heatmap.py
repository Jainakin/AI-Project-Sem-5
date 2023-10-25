import ImportAndClean

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

# plt.style.use('seaborn')

crimes = ImportAndClean.cleaning()
days = ['Monday','Tuesday','Wednesday','Thursday', 'Friday', 'Saturday', 'Sunday']

hour_by_location = crimes.pivot_table(values='ID', index='Location Description', columns=crimes.index.hour, aggfunc=np.size).fillna(0)
hour_by_type     = crimes.pivot_table(values='ID', index='Primary Type', columns=crimes.index.hour, aggfunc=np.size).fillna(0)
hour_by_week     = crimes.pivot_table(values='ID', index=crimes.index.hour, columns=crimes.index.day_name(), aggfunc=np.size).fillna(0)
hour_by_week     = hour_by_week[days].T # just reorder columns according to the the order of days
dayofweek_by_location = crimes.pivot_table(values='ID', index='Location Description', columns=crimes.index.dayofweek, aggfunc=np.size).fillna(0)
dayofweek_by_type = crimes.pivot_table(values='ID', index='Primary Type', columns=crimes.index.dayofweek, aggfunc=np.size).fillna(0)
location_by_type  = crimes.pivot_table(values='ID', index='Location Description', columns='Primary Type', aggfunc=np.size).fillna(0)

from sklearn.cluster import AgglomerativeClustering as AC

def scale_df(df,axis=0):
    '''
    A utility function to scale numerical values (z-scale) to have a mean of zero
    and a unit variance.
    '''
    return (df - df.mean(axis=axis)) / df.std(axis=axis)

def plot_hmap(df, ix=None, cmap=plt.colormaps['plasma']):
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
    Calculates the scaled values within each row of df and plot_hmap
    '''
    df_marginal_scaled = scale_df(df.T).T
    if ix is None:
        ix = AC(4).fit(df_marginal_scaled).labels_.argsort() 
    cap = np.min([np.max(df_marginal_scaled.values), np.abs(np.min(df_marginal_scaled.values))])
    df_marginal_scaled = np.clip(df_marginal_scaled, -1*cap, cap)
    plot_hmap(df_marginal_scaled, ix=ix)
    
def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

plt.figure(figsize=(10,9))
scale_and_plot(hour_by_type)

# predicting most likely location for a certain type of crime
df = normalize(location_by_type)
ix = AC(3).fit(df.T).labels_.argsort() 

plt.figure(figsize=(15,12))
plt.imshow(df.T.iloc[ix,:], cmap=plt.colormaps['Reds'])
plt.colorbar(fraction=0.03)
plt.xticks(np.arange(df.shape[0]), df.index, rotation='vertical')
plt.yticks(np.arange(df.shape[1]), df.columns)
plt.title('Normalized values for frequency of location for each crime')
plt.xlabel("Time of day in a 24 hour format")
plt.ylabel("Crimes")
plt.grid(False)
plt.show()

# Visually translating matrix into Heatmap
#initialization of location bounds
crimes.iloc[(crimes[['Longitude']].values < -88.0).flatten(), crimes.columns=='Longitude'] = 0.0
crimes.iloc[(crimes[['Longitude']].values > -87.5).flatten(), crimes.columns=='Longitude'] = 0.0
crimes.iloc[(crimes[['Latitude']].values < 41.60).flatten(),  crimes.columns=='Latitude'] = 0.0
crimes.iloc[(crimes[['Latitude']].values > 42.05).flatten(),  crimes.columns=='Latitude'] = 0.0
crimes.replace({'Latitude': 0.0, 'Longitude': 0.0}, np.nan, inplace=True)
crimes.dropna(inplace=True)#drops null value rows


crimes_new = crimes[(crimes['Primary Type'] == 'THEFT') | (crimes['Primary Type'] == 'CRIMINAL DAMAGE') | (crimes['Primary Type'] == 'BATTERY')]
ax = sns.lmplot(#'Longitude', 'Latitude',
                crimes_new[['Longitude','Latitude']],
                fit_reg=False,
                height=4, 
                scatter_kws={'alpha':.1})
ax = sns.kdeplot(crimes_new[['Longitude','Latitude']], 
                # cmp = plt.colormaps['inferno'], 
                bw_method=.005, 
                #n_levels=10,
                cbar=True, 
                fill=False, 
                thresh=False)
ax.set_xlim(-87.9,-87.5)
ax.set_ylim(41.60,42.05)
ax.set_axis_off()

ctypes = ['THEFT', 'BATTERY ', 'CRIMINAL DAMAGE']
fig = plt.figure(figsize=(15,35))
for i in ctypes:
    ax = fig.add_subplot(int(np.ceil(float(len(ctypes)) / 4)), 4, i+1)
    crimes_ = crimes[crimes['Primary Type']==ctypes]
    sns.regplot('Longitude', 'Latitude', data= crimes_[['Longitude','Latitude']], fit_reg=False, scatter_kws={'alpha':.1, 'color':'grey'}, ax=ax)
    sns.kdeplot(X='Longitude', Y='Latitude', data= crimes_[['Longitude','Latitude']], 
                #cmap=plt.colormaps['inferno'], 
                bw_method=.005, n_levels=10,
                cbar=True, fill=True, thresh=False, ax = ax)
    ax.set_title(ctypes)
    ax.set_xlim(-87.9,-87.5)
    ax.set_ylim(41.60,42.05)
    ax.set_axis_off()    
    plt.show()