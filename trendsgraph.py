import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
#plt.style.use('seaborn')
import seaborn as sns
import ImportAndClean

def trends():
    #crimes = pd.read_csv("C:\\Users\\suran\\Desktop\\School\\1 UNIVERSITY\\BENNETT\\CSET211 Ai\\AI Hackathon\\Chicago_Crimes_2012_to_2017\\Chicago_Crimes_2012_to_2017_trimmed.csv")
    crimes = ImportAndClean.cleaning()
    # print('working')
    # print('Dataset Shape before drop_duplicate : ', crimes.shape)
    #crimes.drop_duplicates(subset=['ID', 'Case Number'], inplace=True)
    # print('Dataset Shape after drop_duplicate: ', crimes.shape)

    #crimes.drop(['Unnamed: 0', 'Case Number','Updated On','Year', 'FBI Code', 'Beat','Ward','Community Area', 'Location'], inplace=True, axis=1)

    #print(crimes.head(3))

    #crimes.Date = pd.to_datetime(crimes.Date, format='%m/%d/%Y %I:%M:%S %p', errors = 'ignore')
    #crimes.index = pd.DatetimeIndex(crimes.Date)

    print(crimes.shape, "\n", crimes.info())

 # Visualisation

# IUCR_data = crimes[crimes["IUCR"] == "0820" or crimes["IUCR"] == "0810"]
# IUCR_data = crimes.query('IUCR  == "0820" or IUCR == "0810"') # Theft
# IUCR_data = crimes.query('IUCR  == "0460" or IUCR == "0486"') # Battery
# IUCR_data = crimes.query('IUCR  == "1320"') # Criminla Damage
    #data_feed = ['IUCR  == "0820" or IUCR == "0810"', 'IUCR  == "0460" or IUCR == "0486"', 'IUCR  == "1320"']
    #data_feed = ImportAndClean.crimes_filt()

    crime_type = ["Theft", "Battery", "Criminal Damage"]
    for i in range(0, 3):
        #IUCR_data = crimes.query(data_feed[i])
        IUCR_data = crimes.query(crime_type[i])
        plt.figure(figsize=(11,5))
        IUCR_data.resample('M').size().plot(legend=False)
        plt.title('Number of {} per month (2012 - 2017)'.format(crime_type[i]))
        plt.xlabel('Months')
        plt.ylabel('Number of {}'.format(crime_type[i]))
        plt.show()
trends()

