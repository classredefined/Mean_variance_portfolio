#%%
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style="darkgrid")
import matplotlib.pyplot as plt
import cvxpy as cp  #pip install cvxpy --upgrade

#%%
# Load the data from yahoo finance.
# Take start date as 2010
import yfinance as yf
AMZN = yf.download('AMZN',start='2010-01-01')

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
stock_namelist=['^GSPC', '^IXIC', '^GDAXI', '^N225', '^FCHI']
ind_data=pd.DataFrame()
for t in stock_namelist:
    ind_data[t]=yf.download(t, start='2010-01-01')['Adj Close']

ind_data.tail(5)
# %%
print(ind_data.iloc[0])

(ind_data/ind_data.iloc[0]).plot(figsize=(16,5))
plt.title('Price change')
plt.xlabel('Date')
plt.ylabel('Percentage')
plt.show
# %%
returns_indi=(ind_data/ind_data.shift(1))-1
returns_indi= returns_indi.dropna()
print(returns_indi.tail(5))
markets_returns=returns_indi.mean()*250
print(markets_returns)

# %%
plt.bar(Indices,list(markets_returns),align='center',alpha=0.5)
plt.ylabel('Value')
plt.title('Annual return')
plt.show()

#%%


# %%
# train test split 100 first samples
T_trn = 100  
ret_mat = np.ascontiguousarray(returns_indi.iloc[1::,:])
equity_trn = ret_mat[:T_trn,]
equity_tst = ret_mat[T_trn:,]

mu_trn = np.mean(equity_trn, axis=0) 
Sigma_trn = np.cov(equity_trn.T)

# Equally weighted portfolio
w_EWP = np.ones(returns_indi.shape[1])/returns_indi.shape[1]
w_EWP
#%%
# Minimum variance portfolio:
# As investors are assumed risk-averse but still want to make a good profit, we introduce the global minimum variance portfolio and the minimum variance portfolio with shortselling contraints

def GMVP(Sigma):
    ones = np.ones(Sigma.shape[0])
    Sigma_inv_1 = np.linalg.solve(Sigma, ones) # same as Sigma_inv @ ones
    w = Sigma_inv_1 / (np.sum(Sigma_inv_1))
    return w
w_GMVP = GMVP(Sigma_trn)
print(w_GMVP)
#%%
def MVP(mu, Sigma, cons ,w_EWP):
    w = cp.Variable(len(mu))
    variance = cp.quad_form(w, Sigma)
    expected_return = w @ mu
    if cons == "no short sell":
        constraint = [w @ mu >= w_EWP @ mu, w >= 0, cp.sum(w) == 1]
    if cons == "short sell":    
        constraint = [w @ mu >= w_EWP @ mu,cp.sum(w) == 1]
    problem = cp.Problem(cp.Minimize(variance), constraint)   
    problem.solve()         
    return w.value
w_MVP_short = MVP(mu_trn, Sigma_trn,"no short sell",w_EWP)
print(w_MVP_short)
#%% Maximum sharpe ratio portfolio:
def MSR(mu, Sigma):
    ones = np.ones(Sigma.shape[0])
    Sigma_inv_1 = np.linalg.solve(Sigma, mu) # same as Sigma_inv @ ones
    w = Sigma_inv_1 / (np.sum(Sigma_inv_1  ))
    return w
w_MSR = MSR(mu_trn, Sigma_trn)

allocation = pd.DataFrame([w_GMVP,w_MVP_short,w_EWP,w_MSR],columns = stock_namelist)
allocation.index = ['GMVP','MVP','EWP','MSR']
allocation.T.plot.bar(figsize = (15,5))
plt.xticks(rotation=0)
plt.show()

print(allocation)

#%%

def performance(returns_indi,allocation):
    ret = []
    vol = []
    sharpe = []
    wealth = []
    for i in range(4):
        ret.append((np.array(returns_indi) @ (allocation.iloc[i,:].T)).mean())
        vol.append(np.sqrt(np.dot(allocation.iloc[i,:],np.dot(returns_indi.cov(),allocation.iloc[i,:].T))))
        sharpe.append(ret[i]/vol[i])
    performance = pd.DataFrame([ret,vol,sharpe],columns = allocation.index).T
    return performance

performance  = pd.concat([performance(returns_indi.iloc[:T_trn,],allocation),
                         performance(returns_indi.iloc[T_trn:,],allocation)], axis = 1)
performance.columns = ['avg excess return train','volatility train','sharpe ratio train',
                       'avg excess return test','volatility test','sharpe ratio test']
performance

#%%

performance.T.plot.bar(figsize = (15,5))
plt.xticks(rotation=0)
plt.show()


#%%
wealth_geom_trn = []
wealth_geom_tst = []
for i in range(4):
    wealth_geom_trn.append(np.cumprod(np.array(returns_indi.iloc[:T_trn,]) @ (allocation.iloc[i,:].T)+1))
    wealth_geom_tst.append(np.cumprod(np.array(returns_indi.iloc[T_trn:,]) @ (allocation.iloc[i,:].T)+1))  

# plots
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15,5))
plt.figure(figsize=(10,15))
ax1.plot(np.array(wealth_geom_trn).T)
ax2.plot(np.array(wealth_geom_tst).T)
fig.suptitle('Portfolios performance (compounded)')
ax1.set_title("in-sample")
ax2.set_title("out-of-sample")
ax1.legend(allocation.index)
ax2.legend(allocation.index)
plt.show()

#%%

# zoom in on the in sample plot:

fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15,5))
plt.figure(figsize=(10,15))
ax1.plot(np.array(wealth_geom_trn).T)
ax2.plot(np.array(wealth_geom_tst).T)
fig.suptitle('Portfolios performance (compounded)')
ax1.set_title("in-sample")
ax2.set_title("out-of-sample")
ax1.legend(allocation.index)
ax2.legend(allocation.index)
ax1.set(ylim=(0,3))
plt.show()

# %%
