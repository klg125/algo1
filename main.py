
import numpy as np #The Numpy numerical computing library
import pandas as pd #The Pandas data science library
import requests #The requests library for HTTP requests in Python
import xlsxwriter #The XlsxWriter library for 
import math #The Python math module
import pyEX as p
from scipy import stats

api='Tpk_c255feee446947889a6e1af8192b50ff'

stocks = pd.read_csv('/Users/kevinli/Documents/sp_500_stocks.csv')

my_columns = ['Ticker', 'Price','Stochastic Indicators', "ST Percentile", 'RSI Score',
              'RSI Percentile' , 'Growth Score', '5 Day Change', '5 Day Percentile',
              '30 Day Change', '30 Day Percentile', '3 Month Change', '3 Month Percentile',
              'Momentum Score', 'PE Ratio', 'PE Percentile', 'PB Ratio', 'PB Percentile',
              'EV/EBITDA', 'EV Percentile', 'Value Score', 'Final Score']
final_dataframe = pd.DataFrame(columns = my_columns)

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]   
        
symbol_groups = list(chunks(stocks['Ticker'], 100))
symbol_strings = []

for i in range(0, len(symbol_groups)):
    symbol_strings.append(','.join(symbol_groups[i]))

for symbol_string in symbol_strings:
#     print(symbol_strings)
    batch_api_call_url = f'https://sandbox.iexapis.com/stable/stock/market/batch/?types=indicator,advanced-stats,quote&symbols={symbol_string}&token={api}'
    data = requests.get(batch_api_call_url).json()

    for symbol in symbol_string.split(','):
        enterprise_value = data[symbol]['advanced-stats']['enterpriseValue']
        ebitda = data[symbol]['advanced-stats']['EBITDA']
        try:
            ev_to_ebitda = enterprise_value/ebitda
        except TypeError:
            ev_to_ebitda = np.NaN

        final_dataframe = final_dataframe.append(
                                        pd.Series([symbol,
                data[symbol]['quote']['latestPrice'],
                data[symbol]['indicator']['rsi'],
                'N/A',
                'N/A',
                'N/A',
                'N/A',
                data[symbol]['advanced-stats']['day5ChangePercent'],
                'N/A',
                data[symbol]['advanced-stats']['day30ChangePercent'],
                'N/A',
                data[symbol]['advanced-stats']['month3ChangePercent'],
                'N/A',
                'N/A',
                data[symbol]['quote']['peRatio'],
                'N/A',
                data[symbol]['advanced-stats']['priceToBook'],
                'N/A',
                ev_to_ebitda,
                'N/A',
                'N/A',
                'N/A',
                                                    ], 
                                                  index = my_columns), 
                                        ignore_index = True)
        

for sym in stocks['Ticker']:
    #RSI Score
    rsi=p.technicalsDF(sym, indicator='rsi', token=api)
    rsi=rsi['rsi']
    rsi=rsi[len(rsi)-5:len(rsi)]
    avg_rsi_score=100-np.mean(rsi)
    final_dataframe.loc['row_count', 'RSI Score']=avg_rsi_score
    
    #Stochastic 
    stoch=p.technicalsDF(sym, indicator='stoch', token=api)
    stoch=stoch['stock_k'][-5:]
    avg_stoch_score=100-np.mean(stoch)
    final_dataframe.loc['row_count','Stochastic Indicator']=avg_stoch_score
    row_count+=1
    print(row_count)


for column in ['PE Ratio', 'PB Ratio', 'EV/Revenue']:
    final_dataframe[column].fillna(final_dataframe[column].mean(), inplace = True)

metrics = {
            'Stochastic Indicators' : "ST Percentile",
            'RSI Score': "RSI Percentile",
            '5 Day Change': "5 Day Percentile",
            '30 Day Change':'30 Day Percentile',
            '3 Month Change':'3 Month Percentile',
            'PE Ratio': 'PE Percentile',
            'PB Ratio': 'PB Percentile', 
            'EV/Revenue': 'EV Percentile'
                
}

for row in final_dataframe.index:
    for metric in metrics.keys():
        final_dataframe.loc[row, metrics[metric]] = stats.percentileofscore(final_dataframe[metric], final_dataframe.loc[row, metric])


#Calculating Growth Score
for row in final_dataframe.index:
    final_dataframe.loc[row, 'Growth Score']=0.5*final_dataframe.loc[row, 'ST Percentile']+0.5*final_dataframe.loc[row, 'RSI Percentile']
    final_dataframe.loc[row, 'Momentum Score']=0.5*final_dataframe.loc[row, '3 Month Percentile']
    +0.3*final_dataframe.loc[row, '30 Day Percentile']+0.2*final_dataframe.loc[row, '5 Day Percentile']
    final_dataframe.loc[row, 'Value Score']=0.33*final_dataframe.loc[row, 'PE Percentile'] + 0.33*final_dataframe.loc[row, 'PB Percentile']
    +0.33*final_dataframe.loc[row, 'EV Percentile']
    final_dataframe.loc[row, 'Final Score']=0.5*final_dataframe.loc[row, 'Growth Score']
    +0.2*final_dataframe.loc[row, 'Momentum Score']+0.3*final_dataframe.loc[row, 'Value Score']
    

#Sorting Final Scores

final_dataframe.sort_values(by = 'Final Score', inplace = True)
