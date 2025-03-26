import pandas as pd
import numpy as np
from statsmodels.api import OLS
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf

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

NSIMS = 1000
NDISPLAYS = 5
np.random.seed(0)
dfPrice = pd.read_excel('century.xlsx', sheet_name = 'price')
vol = dfPrice['Volatility'].values[1:]
price = dfPrice['Price'].values
dividend = dfPrice['Dividends'].values
dfEarnings = pd.read_excel('century.xlsx', sheet_name = 'earnings')
earnings = dfEarnings['Earnings'].values
N = len(vol)
L = 9

def gaussNoise(data, size):
    covMatrix = data.cov()
    simNoise = []
    dim = len(data.keys())
    for sim in range(NSIMS):
        simNoise.append(np.transpose(np.random.multivariate_normal(np.zeros(dim), covMatrix, size)))
    return simNoise

def KDE(data, size):
    method = stats.gaussian_kde(np.transpose(data.values), bw_method = 'silverman')
    simNoise = []
    for sim in range(NSIMS):
        simNoise.append(np.array(method.resample(size)))
    return simNoise

lvol = np.log(vol)
RegVol = stats.linregress(lvol[:-1], lvol[1:])
betaVol = RegVol.slope
alphaVol = RegVol.intercept
residVol = [lvol[k+1] - betaVol * lvol[k] - alphaVol for k in range(N-1)]
meanVol = np.mean(vol)

def fit(model, window, inflMode):
    VolFactors = pd.DataFrame({'const' : 1/vol, 'vol' : 1})
    if inflMode == 'Nominal':
        index = price
        div = dividend
        total = np.array([np.log(index[k+1] + dividend[k+1]) - np.log(index[k]) for k in range(N)])
        if model == 'EarningsYield':
            earn = earnings
    if inflMode == 'Real':
        cpi = dfEarnings['CPI'].values
        index = cpi[-1]*price/cpi[L:]
        div = cpi[-1]*dividend/cpi[L:]
        total = np.array([np.log(index[k+1] + div[k+1]) - np.log(index[k]) for k in range(N)])
        if model == 'EarningsYield':
            earn = cpi[-1]*earnings/cpi
    Nprice = np.diff(np.log(index))/vol
    Ntotal = total/vol
    if model == 'Simple':
        Model = OLS(Ntotal, VolFactors).fit()
        res = Model.resid
        allResids = pd.DataFrame({'Volatility' : residVol, 'Returns' : res[1:]})
        dictRegressions = {'Returns': Model}
        dictMeans = {'Volatility': round(meanVol, 1)}
        dictCurrents = {'Volatility' : vol[-1]}
    if model == 'EarningsYield':
        cumearn = np.array([np.mean(earn[k-window:k]) for k in range(L + 1, L + N + 2)])
        earnyield = cumearn/index
        plt.plot(range(1928, 1929 + N), earnyield)
        plt.title('Cyclically Adjusted Earnings Yield Averaged over ' + str(window) + ' Years')
        plt.savefig('earnings-yield.png')
        plt.close()
        earngr = np.diff(np.log(earn[L:]))
        growth = earngr/vol
        ModelGrowth = OLS(growth, VolFactors).fit()
        resGrowth = ModelGrowth.resid
        AllFactors = VolFactors
        AllFactors['yield'] = np.log(earnyield[:-1])/vol
        ModelPrice = OLS(Nprice, AllFactors).fit()
        resPrice = ModelPrice.resid
        Model = OLS(Ntotal, AllFactors).fit()
        res = Model.resid
        allResids = pd.DataFrame({'Volatility':residVol, 'Growth': resGrowth[1:], 'Price' : resPrice[1:], 'Returns' : res[1:]})
        dictRegressions = {'Returns' : Model, 'Growth' : ModelGrowth, 'Price' : ModelPrice}
        dictMeans = {'Volatility' : round(meanVol, 1), 'EarningsYield' : round(np.mean(earnyield), 3)}
        dictCurrents = {'Volatility' : vol[-1], 'Earnings' : earn[-window:], 'Price' : index[-1]}
    for resid in allResids:
        plots(allResids[resid], resid)
    return allResids, dictRegressions, dictMeans, dictCurrents
   
def simReturns(model, window, inflMode, residMode, horizon):
    allResids, allModels, allMeans, allCurrents = fit(model, window, inflMode)
    if residMode == 'Gauss':
        innovations = gaussNoise(allResids, horizon)
    if residMode == 'KDE':
        innovations = KDE(allResids, horizon)
    allSims = []
    for sim in range(NSIMS):
        simRet = []
        simLVol = [np.log(allCurrents['Volatility'])]
        innovation = innovations[sim]
        for t in range(horizon):
            simLVol.append(simLVol[-1]*betaVol + alphaVol + innovation[0, t])
        simVol = np.exp(simLVol)
        if model == 'Simple':
            simRet = [simVol[t+1] * (allModels['Returns'].predict([1/simVol[t+1], 1])[0] + innovation[1, t]) for t in range(horizon)]
            current = {'Volatility' : round(simVol[0], 1)}
        if model == 'EarningsYield':
            simGrowth = [simVol[t+1] * (allModels['Growth'].predict([1/simVol[t+1], 1])[0] + innovation[1, t]) for t in range(horizon)]
            oldEarn = allCurrents['Earnings']
            initialYield = np.mean(oldEarn)/allCurrents['Price']
            simEarn = np.append(oldEarn, oldEarn[-1] * np.exp(np.cumsum(simGrowth)))
            simPrice = [allCurrents['Price']]
            simEarnYield = [initialYield]
            current = {'Volatility' : round(simVol[0], 1), 'EarningsYield' : round(initialYield, 3)}
            for t in range(horizon):
                simPriceRet = simVol[t+1] * (allModels['Price'].predict([1/simVol[t+1], 1, np.log(simEarnYield[t])/simVol[t+1]])[0] + innovation[2, t])
                simPrice.append(simPrice[t] * np.exp(simPriceRet))
                simEarnYield.append(np.mean(simEarn[t+1:t+1+window])/simPrice[t+1])
            simRet = [simVol[t+1] * (allModels['Returns'].predict([1/simVol[t+1], 1, np.log(simEarnYield[t])/simVol[t+1]])[0] + innovation[3, t]) for t in range(horizon)]
        allSims.append(simRet)
    return np.array(allSims), allMeans, current

def output(model, window, inflMode, residMode, horizon, initialWealth, flow):
    if flow == 0:
        flowText = 'No regular contributions or withdrawals'
    if flow > 0:
        flowText = 'Contributions ' + str(flow) + ' per year'
    if flow < 0:
        flowText = 'Withdrawals ' + str(abs(flow)) + ' per year'
    paths = []
    timeAvgRets = []
    simulatedReturns, allMeans, allCurrents = simReturns(model, window, inflMode, residMode, horizon)
    for sim in range(NSIMS):
        path = [initialWealth]
        simReturn = simulatedReturns[sim]
        timeAvgRets.append(np.mean(simReturn))
        for t in range(horizon):
            if (path[t] == 0):
                path.append(0)
            else:
                new = max(path[t] * np.exp(simReturn[t]) + flow, 0)
                path.append(new)
        paths.append(path)
    paths = np.array(paths)
    avgRet = np.mean([timeAvgRets[sim] for sim in range(NSIMS) if paths[sim, -1] > 0])
    wealthMean = np.mean(paths[:, -1])
    meanProb = np.mean([paths[sim, -1] > wealthMean for sim in range(NSIMS)])
    ruinProb = np.mean([paths[sim, -1] == 0 for sim in range(NSIMS)])
    sortedIndices = np.argsort(paths[:, -1])
    selectedIndices = [sortedIndices[int(NSIMS*(2*k+1)/(2*NDISPLAYS))] for k in range(NDISPLAYS)]
    times = range(horizon + 1)
    simText = str(NSIMS) + ' of simulations'
    timeHorizonText = 'Time Horizon: ' + str(horizon) + ' years'
    inflText = inflMode + ' returns'
    initWealthText = 'Initial Wealth ' + str(round(initialWealth))
    Portfolio = 'The portfolio: 100% Large Stocks'
    if model == 'Simple':
        modelText = 'Modeling using only volatility'
    if model == 'EarningsYield':
        modelText = 'Modeling using volatility and earnings yield with trailing ' + str(window) + ' years'
    initMarketText = 'Initial (Current) conditions: ' + ' '.join([key + ' ' + str(allCurrents[key]) for key in allCurrents])
    avgMarketText = 'Historical averages: ' + ' '.join([key + ' ' + str(allMeans[key]) for key in allMeans])
    SetupText = 'SETUP: ' + simText + '\n' + modelText + '\n' + Portfolio + '\n' + timeHorizonText + '\n' + inflText + '\n' + initWealthText +'\n' + initMarketText + '\n' + avgMarketText + '\n' + flowText + '\n'
    if np.isnan(avgRet):
        ResultText = 'RESULTS: 100% Ruin Probability, always zero wealth'
    else:
        RuinProbText = str(round(100*ruinProb, 2)) + '% Ruin Probability'
        AvgRetText = 'time averaged annual returns:\naverage over all paths without ruin ' + str(round(100*avgRet, 2)) + '%'
        MeanText = 'average final wealth ' + str(round(wealthMean))
        MeanCompText = 'final wealth exceeds average with probability ' + str(round(100*meanProb, 2)) + '%'
        ResultText = 'RESULTS: ' + RuinProbText + '\n' + AvgRetText + '\n' + MeanText + '\n' + MeanCompText
    bigTitle = SetupText + '\n' + ResultText + '\n'
    plt.plot([0, 4000], color = 'w', label = bigTitle)
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

output('EarningsYield', 5, 'Real', 'KDE', 30, 1000, -40)