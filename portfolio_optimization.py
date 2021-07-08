#%%
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style="darkgrid")
import matplotlib.pyplot as plt
import cvxpy as cp  #pip install cvxpy --upgrade
import yfinance as yf
#%% Multi-asset analysis
#Calculate returns of individual stocks
tickers=['AAPL','FB','AMZN','BAC','GE','MSFT','TSLA','NFLX']
df=pd.DataFrame()
for t in tickers:
    df[t]=yf.download(t, start='2010-01-01')['Adj Close']

df.head(5)
# %%
print(df.iloc[0])

(df/df.iloc[0]).plot(figsize=(16,5))
plt.title('Price change')
plt.xlabel('Date')
plt.ylabel('Percentage')
plt.show
# %%
returns=(df/df.shift(1))-1
returns= returns.dropna()
print(returns.tail(5))
daily_returns=returns.mean()*250
print(daily_returns)

plt.bar(tickers,list(daily_returns),align='center',alpha=0.5)
plt.ylabel('Value')
plt.title('Annual return')
plt.show()

# %%
# train test split 70-30
T_trn = round(len(returns)*.7)
ret_mat = np.ascontiguousarray(returns.iloc[1::,:])
ret_trn = ret_mat[:T_trn,]
ret_tst = ret_mat[T_trn:,]

mu_trn = np.mean(ret_trn, axis=0) 
sigma_trn = np.cov(ret_trn.T)
print(mu_trn,sigma_trn)
# Equally weighted portfolio
w_EWP = np.ones(returns.shape[1])/returns.shape[1]
w_EWP
#%% Minimum variance portfolio:
# As investors are assumed risk-averse but still want to make a good profit
# we introduce the global minimum variance portfolio and the minimum variance portfolio with shortselling contraints

def GMVP(Sigma):
    ones = np.ones(Sigma.shape[0])
    Sigma_inv_1 = np.linalg.solve(Sigma, ones) # same as Sigma_inv @ ones
    w = Sigma_inv_1 / (np.sum(Sigma_inv_1))
    return w
w_GMVP = GMVP(sigma_trn)
print(w_GMVP)

# Minimum variance portfolio with short-selling
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
w_MVP_short = MVP(mu_trn, sigma_trn,"short sell",w_EWP)
print(w_MVP_short)
#%% Maximum sharpe ratio portfolio
# This portfolio show the optimized return/risks ratios
def MSR(mu, Sigma):
    ones = np.ones(Sigma.shape[0])
    Sigma_inv_1 = np.linalg.solve(Sigma, mu) # same as Sigma_inv @ ones
    w = Sigma_inv_1 / (np.sum(Sigma_inv_1  ))
    return w
w_MSR = MSR(mu_trn, sigma_trn)
print(w_MSR)

#%%
allocation = pd.DataFrame([w_GMVP,w_MVP_short,w_EWP,w_MSR],columns = tickers)
allocation.index = ['GMVP','MVP','EWP','MSR']
allocation.T.plot.bar(figsize = (15,5))
plt.xticks(rotation=0)
plt.show()

print(allocation)

#%%
def performance(returns,allocation):
    ret = []
    vol = []
    sharpe = []
    wealth = []
    for i in range(4):
        ret.append((np.array(returns) @ (allocation.iloc[i,:].T)).mean())
        vol.append(np.sqrt(np.dot(allocation.iloc[i,:],np.dot(returns.cov(),allocation.iloc[i,:].T))))
        sharpe.append(ret[i]/vol[i])
    performance = pd.DataFrame([ret,vol,sharpe],columns = allocation.index).T
    return performance

performance  = pd.concat([performance(returns.iloc[:T_trn,],allocation),
                         performance(returns.iloc[T_trn:,],allocation)], axis = 1)
performance.columns = ['avg excess return train','volatility train','sharpe ratio train',
                       'avg excess return test','volatility test','sharpe ratio test']
print(performance)

#%%

performance.T.plot.bar(figsize = (15,5))
plt.xticks(rotation=0)
plt.show()

#%%
wealth_geom_trn = []
wealth_geom_tst = []
for i in range(4):
    wealth_geom_trn.append(np.cumprod(np.array(returns.iloc[:T_trn,]) @ (allocation.iloc[i,:].T)+1))
    wealth_geom_tst.append(np.cumprod(np.array(returns.iloc[T_trn:,]) @ (allocation.iloc[i,:].T)+1))  

# plots
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15,5))
plt.figure(figsize=(10,15))
ax1.plot(np.array(wealth_geom_trn).T)
ax2.plot(np.array(wealth_geom_tst).T)
fig.suptitle('Portfolios performance (compounded)')
ax1.set_title("in-sample")
ax2.set_title("out-of-sample")
ax1.legend(allocation.index,loc = 2)
ax2.legend(allocation.index,loc = 2)
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
ax1.legend(allocation.index,loc = 2)
ax2.legend(allocation.index,loc = 2)
ax1.set(ylim=(0,10))
plt.show()

#%%
'''
MSR:
As we can see that, the MSR allow short-selling the GE stocks, 
which results in significant return in train set
Although it shows the best performance in test set, we can see that the
the performance dropped dramatically for MSR portfolio
This portfolio is good for risk seekers, not risk adverse investors 

GMVP:
This portfolio shows consistency in both train and test dataset
As can be seen from the chart, GMVP show lowest volatility but also lowest returns
This portfolio is the safest bet for risk-adverse investors

MVP:
For this portfolio, we set the expected return to be equal to EWP portfolio.
Although we did include the short-selling option in building this portfolio,
the allocation doesn't have short position.
This portfolio shows higher return than GMVP still with a certain low volatility
In theory the MVP is risk-return optimal for risk-adverse investors.

EWP:
This portfolio is a naive way to build a portoflio.
It isn't belong to Mean-variance framework or Markowitz framework and doesn't optimize any criteria.
However, sometimes this portfolio yields good results, as in this case with this particular dataset of growing companies.

'''

# %%
