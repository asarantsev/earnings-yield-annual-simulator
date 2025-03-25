import pandas as pd
import numpy as np
from statsmodels.api import OLS
import matplotlib.pyplot as plt
from scipy import stats

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

def gaussNoise(data, dim, size):
    covMatrix = np.cov(data)
    simNoise = []
    for sim in range(NSIMS):
        simNoise.append(np.transpose(np.random.multivariate_normal(np.zeros(dim), covMatrix, size)))
    return simNoise

def KDE(data, size):
    method = stats.gaussian_kde(data, bw_method = 'silverman')
    simNoise = []
    for sim in range(NSIMS):
        simNoise.append(np.array(method.resample(size)))
    return simNoise

lvol = np.log(vol)
RegVol = stats.linregress(lvol[:-1], lvol[1:])
betaVol = RegVol.slope
alphaVol = RegVol.intercept
residVol = [lvol[k+1] - betaVol * lvol[k] - alphaVol for k in range(N-1)]
allInflModes = ['Nominal', 'Real']
allModels = {'Simple': ['volatility'], 'EarningsYield': ['volatility', 'earnings yield']}
allDims = {'Simple': 2, 'EarningsYield' : 4}
VolFactors = pd.DataFrame({'const' : 1/vol, 'vol' : 1})

def fit(model, inflMode):
    if inflMode == 'Nominal':
        total = np.array([np.log(price[k+1] + dividend[k+1]) - np.log(price[k]) for k in range(N)])
        if model == 'EarningsYield':
            Nprice = np.diff(np.log(price))/vol
            earngr = np.diff(np.log(earnings))
    if inflMode == 'Real':
        total = np.array([np.log(price[k+1] + dividend[k+1]) - np.log(price[k]) for k in range(N)]) - inflation
        if model == 'EarningsYield':
            Nprice = (np.diff(np.log(price)) - inflation)/vol
            earngr = np.diff(np.log(earnings)) - inflation    
    Ntotal = total/vol
    if model == 'Simple':
        Model = OLS(Ntotal, VolFactors).fit()
        res = Model.resid
        allResids = np.stack([residVol, res[1:]])
        dictionary = {'Returns': Model}
    if model == 'EarningsYield':  
        earnyield = earnings/price
        growth = earngr/vol
        ModelGrowth = OLS(growth, VolFactors).fit()
        resGrowth = ModelGrowth.resid
        AllFactors = VolFactors
        AllFactors['yield'] = earnyield[:-1]/vol
        ModelPrice = OLS(Nprice, AllFactors).fit()
        resPrice = ModelPrice.resid
        Model = OLS(Ntotal, AllFactors).fit()
        res = Model.resid
        allResids = np.stack([residVol, resGrowth[1:], resPrice[1:], res[1:]])
        dictionary = {'Returns' : Model, 'Growth' : ModelGrowth, 'Price' : ModelPrice}    
    return allResids, dictionary
   
def simReturns(model, inflMode, residMode, initialConditions, horizon):
    allResids, allModels = fit(model, inflMode)
    if residMode == 'Gauss':
        innovations = gaussNoise(allResids, allDims[model], horizon)
    if residMode == 'KDE':
        innovations = KDE(allResids, horizon)
    allSims = []
    for sim in range(NSIMS):
        simRet = []
        simLVol = [np.log(initialConditions['Volatility'])]
        innovation = innovations[sim]
        for t in range(horizon):
            simLVol.append(simLVol[-1]*betaVol + alphaVol + innovation[0, t])
        simVol = np.exp(simLVol)
        if model == 'Simple':
            simRet = [simVol[t+1] * (allModels['Returns'].predict([1/simVol[t+1], 1])[0] + innovation[1, t]) for t in range(horizon)]
        if model == 'EarningsYield':
            simGrowth = [simVol[t+1] * (allModels['Growth'].predict([1/simVol[t+1], 1])[0] + innovation[1, t]) for t in range(horizon)]
            simEarn = initialConditions['Yield'] * np.append(np.array([1]), np.exp(np.cumsum(simGrowth)))
            simPrice = [1]
            simEarnYield = [initialConditions['Yield']]
            for t in range(horizon):
                simPriceRet = simVol[t+1] * (allModels['Price'].predict([1/simVol[t+1], 1, simEarnYield[t]/simVol[t+1]])[0] + innovation[2, t])
                simPrice.append(simPrice[t] * np.exp(simPriceRet))
                simEarnYield.append(simEarn[t+1]/simPrice[t+1])
            simRet = [simVol[t+1] * (allModels['Returns'].predict([1/simVol[t+1], 1, simEarnYield[t]/simVol[t+1]])[0] + innovation[3, t]) for t in range(horizon)]
        allSims.append(simRet)
    return np.array(allSims)

def output(model, inflMode, residMode, initialConditions, horizon, initialWealth, flow):
    if flow == 0:
        flowText = 'No regular contributions or withdrawals'
    if flow > 0:
        flowText = 'Contributions ' + str(flow) + ' per year'
    if flow < 0:
        flowText = 'Withdrawals ' + str(abs(flow)) + ' per year'
    paths = []
    timeAvgRets = []
    simulatedReturns = simReturns(model, inflMode, residMode, initialConditions, horizon)
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
    initMarketText = 'Initial conditions: ' + ' '.join([key + str(initialConditions[key]) for key in initialConditions])
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
    plt.plot([0], [initialWealth], color = 'w', label = bigTitle)
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

output('EarningsYield', 'Real', 'Gauss', {'Volatility': 10, 'Yield' : 0.07}, 10, 1000, -70)