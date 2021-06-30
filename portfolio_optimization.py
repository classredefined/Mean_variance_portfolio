#%%
import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import seaborn as sns
sns.set(style="darkgrid")
import matplotlib.pyplot as plt

#%%
# Load the data from yahoo finance.
# Take start date as 2010

AMZN=wb.DataReader('AMZN', data_source='yahoo',start='2010-1-1') 

#Check loaded data.
print(AMZN.info())
print(AMZN.tail(5))
print(AMZN.head(5))
plt.plot(AMZN["Adj Close"])


#AMZN[AMZN.Date == "2010-01-04"]
print(AMZN.iloc[0,:])
print(AMZN.iloc[-1,:])
print(AMZN.iloc[-1,5]/AMZN.iloc[0,5]-1)

# Use the adjusted price as the market convention to calculate the return
AMZN['Simple Return']=(AMZN['Adj Close']/AMZN['Adj Close'].shift(1))-1

# Check new column data of simple return
print(AMZN['Simple Return'].head(5))
#%%
# Plot the simple return, this is just the daily rate, not too much meaningful
AMZN['Simple Return'].plot(figsize=(10,8))
plt.title('Simple rate of return')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()


#%%
# Obtain the average daily simple return over 2010-2018
avg_return_d=AMZN['Simple Return'].mean()
print(avg_return_d)

# Obtain the average annual simple return 
# Multiple by 250 trading days over a year
avg_return_a=AMZN['Simple Return'].mean()*250
print(avg_return_a)

print(str(round(avg_return_a,5)*100), "%")
# %%
# Log return
AMZN['Log Return']=np.log(AMZN['Adj Close']/AMZN['Adj Close'].shift(1))
AMZN['Log Return'].plot(figsize=(16,9))
plt.title('Log rate of return')
plt.xlabel('Date')
plt.ylabel('Value')

plt.show()
# %%
log_return_avg=AMZN['Log Return'].mean()
print(log_return_avg)

log_return_Annual_Avg=log_return_avg *250
print(log_return_Annual_Avg)

print(str(round(log_return_Annual_Avg,5)*100),' %')
#%% Multi-asset analysis
#CALCULATE RETURN OF INDICES
Indices=['S&P500','NASDAD','DAX30','NEKKEI','CAC40']
tickers_indices=['^GSPC', '^IXIC', '^GDAXI', '^N225', '^FCHI']
ind_data=pd.DataFrame()
for t in tickers_indices:
    ind_data[t]=wb.DataReader(t,data_source='yahoo', start='2010-01-01')['Adj Close']

ind_data.tail(5)
# %%
ind_data.iloc[0]
# %%
(ind_data/ind_data.iloc[0]).plot(figsize=(16,5))
plt.title('Price change')
plt.xlabel('Date')
plt.ylabel('Percentage')
plt.show
# %%
returns_indi=(ind_data/ind_data.shift(1))-1
returns_indi.tail(5)
markets_returns=returns_indi.mean()*250
markets_returns
# %%
plt.bar(Indices,list(markets_returns),align='center',alpha=0.5)
plt.ylabel('Value')
plt.title('Annual return')
plt.show()

# %%
