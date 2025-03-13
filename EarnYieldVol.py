import pandas as pd
import numpy as np
from statsmodels.api import OLS
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from scipy import stats

def plots(data, label):
    plot_acf(data, zero = False)
    plt.title(label + '\n ACF for Original Values')
    plt.savefig('O-' + label + '.png')
    plt.close()
    plot_acf(abs(data), zero = False)
    plt.title(label + '\n ACF for Absolute Values')
    plt.savefig('A-' + label + '.png')
    plt.close()
    qqplot(data, line = 's')
    plt.title(label + '\n Quantile-Quantile Plot vs Normal')
    plt.savefig('QQ-' + label + '.png')
    plt.close()
    
def analysis(data, label):
    print(label + ' analysis of residuals normality')
    print('Skewness:', stats.skew(data))
    print('Kurtosis:', stats.kurtosis(data))
    print('Shapiro-Wilk p = ', stats.shapiro(data)[1])
    print('Jarque-Bera p = ', stats.jarque_bera(data)[1])

NSIMS = 1000
NDISPLAYS = 5
np.random.seed(0)
dfPrice = pd.read_excel('century.xlsx', sheet_name = 'price')
vol = dfPrice['Volatility'].values[1:]
price = dfPrice['Price'].values
dividend = dfPrice['Dividends'].values
dfEarnings = pd.read_excel('century.xlsx', sheet_name = 'earnings')
earnings = dfEarnings['Earnings'].values[9:]
cpi = dfEarnings['CPI'].values[9:]
inflation = np.diff(np.log(cpi))
N = len(vol)
print(N)
nominalPrice = np.diff(np.log(price))
realPrice = nominalPrice - inflation
nominalTotal = np.array([np.log(price[k+1] + dividend[k+1]) - np.log(price[k]) for k in range(N)])
realTotal = nominalTotal - inflation
lvol = np.log(vol)
RegVol = stats.linregress(lvol[:-1], lvol[1:])
betaVol = RegVol.slope
alphaVol = RegVol.intercept
print('Slope = ', round(betaVol, 3))
print('Intercept = ', round(alphaVol, 3))
residVol = np.array([lvol[k+1] - betaVol * lvol[k] - alphaVol for k in range(N-1)])
plots(residVol, 'AR(1) Volatility Residuals')
analysis(residVol, 'AR(1) Volatility Residuals')
nearngr = np.diff(np.log(earnings))
rearngr = nearngr - inflation
earnyield = earnings/price
ngrowth = nearngr/vol
plots(ngrowth, 'Normalized Nominal Earnings Growth')
analysis(ngrowth, 'Normalized Nominal Earnings Growth')
nmeangrowth = np.mean(ngrowth)
rgrowth = rearngr/vol
plots(rgrowth, 'Normalized Real Earnings Growth')
analysis(rgrowth, 'Normalized Real Earnings Growth')
rmeangrowth = np.mean(ngrowth)
RegDF = pd.DataFrame({'const' : 1/vol, 'volatility' : 1, 'yield' : earnyield[:-1]/vol})
RegTotalReal = OLS(realTotal/vol, RegDF).fit()
print('\nTotal Real Returns Regression\n')
print(RegTotalReal.summary())
resTotalReal = RegTotalReal.resid
plots(resTotalReal, 'Total Real Residuals')
analysis(resTotalReal, 'Total Real Residuals')
RegTotalNominal = OLS(nominalTotal/vol, RegDF).fit()
print('\nTotal Nominal Returns Regression\n')
print(RegTotalNominal.summary())
resTotalNominal = RegTotalNominal.resid
plots(resTotalNominal, 'Total Nominal Residuals')
analysis(resTotalNominal, 'Total Nominal Residuals')
RegPriceReal = OLS(realPrice/vol, RegDF).fit()
print('\nPrice Real Returns Regression\n')
print(RegPriceReal.summary())
resPriceReal = RegPriceReal.resid
plots(resPriceReal, 'Price Real Residuals')
analysis(resPriceReal, 'Price Real Residuals')
RegPriceNominal = OLS(nominalPrice/vol, RegDF).fit()
print('\nPrice Nominal Returns Regression\n')
print(RegPriceNominal.summary())
resPriceNominal = RegPriceNominal.resid
plots(resPriceNominal, 'Price Nominal Residuals')
analysis(resPriceNominal, 'Price Nominal Residuals')
covReal = np.cov(np.stack([residVol, ngrowth[1:], resPriceReal[1:], resTotalReal[1:]]))
print('covariance for real version')
print(covReal)
covNominal = np.cov(np.stack([residVol, rgrowth[1:], resPriceNominal[1:], resTotalNominal[1:]]))
print('covariance for nominal version')
print(covNominal)

def simReturns(infl, initialVol, initialYield, nYears):
    if infl == 'R':
        mean = rmeangrowth
        cov = covReal
        RegPrice = RegPriceReal
        RegTotal = RegTotalReal
    if infl == 'N':
        mean = nmeangrowth
        cov = covNominal
        RegPrice = RegPriceNominal
        RegTotal = RegTotalNominal
    innov = np.random.multivariate_normal(np.zeros(4), cov, nYears)
    simLVol = [np.log(initialVol)]
    simRet = []
    for t in range(nYears):
        simLVol.append(simLVol[-1]*betaVol + alphaVol + innov[t, 0])
    simVol = np.exp(simLVol)
    simGrowth = np.array([(innov[k, 1] + mean) * simVol[k+1] for k in range(nYears)])
    simYield = np.array([initialYield])
    for t in range(nYears):
        logChange = simGrowth[t] - (RegPrice.predict([1/simVol[t+1], 1, simYield[t]/simVol[t+1]]) + innov[t, 2]) * simVol[t+1]
        simYield = np.append(simYield, simYield[t] * np.exp(logChange))
    simRet = np.array([(RegTotal.predict([1/simVol[t+1], 1, simYield[t]/simVol[t+1]]) + innov[t, 3]) * simVol[t+1] for t in range(nYears)])
    return simYield, simRet

def simWealth(infl, initialV, initialY, initialW, flow, horizon):
    simYield, returns = simReturns(infl, initialV, initialY, horizon)
    timeAvgRet = np.mean(returns)
    wealth = np.array([initialW])
    for t in range(horizon):
        if (wealth[t] == 0):
            wealth = np.append(wealth, [0])
        else:
            new = max(wealth[t] * np.exp(returns[t]) + flow, 0)
            wealth = np.append(wealth, new)
    return timeAvgRet, wealth, simYield

def output(infl, initialV, initialY, initialW, flow, horizon):
    if flow == 0:
        flowText = 'No regular contributions or withdrawals'
    if flow > 0:
        flowText = 'Contributions ' + str(flow) + ' per year'
    if flow < 0:
        flowText = 'Withdrawals ' + str(abs(flow)) + ' per year'
    paths = []
    yields = []
    timeAvgRets = []
    for sim in range(NSIMS):
        timeAvgRet, wealth, simYield = simWealth(infl, initialV, initialY, initialW, flow, horizon)
        timeAvgRets.append(timeAvgRet)
        paths.append(wealth)
        yields.append(simYield)
    paths = np.array(paths)
    yields = np.array(yields)
    avgRet = np.mean([timeAvgRets[sim] for sim in range(NSIMS) if paths[sim, -1] > 0])
    wealthMean = np.mean(paths[:, -1])
    meanProb = np.mean([paths[sim, -1] > wealthMean for sim in range(NSIMS)])
    ruinProb = np.mean([paths[sim, -1] == 0 for sim in range(NSIMS)])
    sortedIndices = np.argsort(paths[:, -1])
    selectedIndices = [sortedIndices[int(NSIMS*(2*k+1)/(2*NDISPLAYS))] for k in range(NDISPLAYS)]
    times = range(horizon + 1)
    simText = str(NSIMS) + ' of simulations'
    timeHorizonText = 'Time Horizon: ' + str(horizon) + ' years'
    if infl == 'N':
        inflText = 'Nominal returns, not inflation-adjusted'
    if infl == 'R':
        inflText = 'Real returns, inflation-adjusted'
    initWealthText = 'Initial Wealth ' + str(round(initialW))
    Portfolio = 'The portfolio: 100% Large Stocks'
    initMarketText = 'Initial conditions: Volatility ' + str(round(initialV, 2)) + ' Earn Yield ' + str(round(100*initialY, 2)) + '%' 
    SetupText = 'SETUP: ' + simText + '\n' + Portfolio + '\n' + timeHorizonText + '\n' + inflText + '\n' + initWealthText +'\n' + initMarketText + '\n' + flowText + '\n'
    if np.isnan(avgRet):
        ResultText = 'RESULTS: 100% Ruin Probability, always zero wealth'
    else:
        RuinProbText = str(round(100*ruinProb, 2)) + '% Ruin Probability'
        AvgRetText = 'time averaged annual returns:\naverage over all paths without ruin ' + str(round(100*avgRet, 2)) + '%'
        MeanText = 'average final wealth ' + str(round(wealthMean))
        MeanCompText = 'final wealth exceeds average with probability ' + str(round(100*meanProb, 2)) + '%'
        ResultText = 'RESULTS: ' + RuinProbText + '\n' + AvgRetText + '\n' + MeanText + '\n' + MeanCompText
    bigTitle = SetupText + '\n' + ResultText + '\n'
    plt.plot([0], [initialW], color = 'w', label = bigTitle)
    for display in range(NDISPLAYS):
        index = selectedIndices[display]
        rankText = ' final wealth, ranked ' + str(round(100*(2*display + 1)/(2*NDISPLAYS))) + '% '
        selectTerminalWealth = round(paths[index, -1])
        if (selectTerminalWealth == 0):
            plt.plot(times, paths[index], label = '0' + rankText + 'Gone Bust !!!')
        else:
            plt.plot(times, paths[index], label = str(selectTerminalWealth) + rankText + 'returns: ' + str(round(100*timeAvgRets[index], 2)) + '%')
    plt.xlabel('Years')
    plt.ylabel('Wealth')
    plt.title('Wealth Plot')
    plt.legend(bbox_to_anchor=(1.05, 1.05), loc='upper left', prop={'size': 12})
    image_path = 'wealth.png'
    plt.savefig(image_path, bbox_inches='tight')
    plt.close()
    for display in range(NDISPLAYS):
        index = selectedIndices[display]
        rankText = ' final wealth ranked ' + str(round(100*(2*display + 1)/(2*NDISPLAYS))) + '% '
        selectTerminalWealth = round(paths[index, -1])
        if (selectTerminalWealth == 0):
            plt.plot(times, yields[index], label = rankText)
        else:
            plt.plot(times, yields[index], label = rankText)
    plt.xlabel('Years')
    plt.ylabel('Yields')
    plt.title('Yield Plot')
    plt.legend(bbox_to_anchor=(1.05, 1.05), loc='upper left', prop={'size': 12})
    image_path = 'yields.png'
    plt.savefig(image_path, bbox_inches='tight')
    plt.close()
    
output(infl = 'R', initialV = 10.5, initialY = 0.07, initialW = 1000, flow = 0, horizon = 20)