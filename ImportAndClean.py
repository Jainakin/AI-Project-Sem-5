
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#plt.style.use('seaborn')


def cleaning():
    crimes = pd.read_csv("C:\\Users\\suran\\Desktop\\School\\1 UNIVERSITY\\BENNETT\CSET204 Prob and Stats\\Hackathon Predictive policing\\Hackathon Predictive policing\\Chicago_Crimes_2012_to_2017.csv\\Chicago_Crimes_2012_to_2017.csv")
    crimes.drop_duplicates(subset=['ID', 'Case Number'], inplace=True)

    crimes.Date = pd.to_datetime(crimes.Date, format='%m/%d/%Y %I:%M:%S %p',errors='ignore')
    crimes.index = pd.DatetimeIndex(crimes.Date)
    # print('Dataset Shape after drop_duplicate: ', crimes.shape)
    crimes.drop(['Unnamed: 0', 'Case Number','Updated On','Year', 'FBI Code', 'Beat','Ward','Community Area', 'Location'], inplace=True, axis=1)


    iucr_codes = ['1320', '0810' , '0820', '0460', '0480']
    crimes = crimes[crimes['IUCR'].isin(iucr_codes)]

    #print(crimes.head(10))
    #print(crimes.info())
    return(crimes)



