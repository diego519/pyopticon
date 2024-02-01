import pandas as pd
import requests
from bs4 import BeautifulSoup
from numpy import floor, nan
from io import StringIO

def rounded_midpoint(bid,ask,base):
    return floor(round((bid+ask)/2,2)/base)*base

def yahoo_option_prices(ticker,date):
    '''Deprecated: Yahoo option prices are significantly delayed
    Use fidelity_option_prices instead'''
    timestamp_date = int(pd.to_datetime(date).timestamp())
    url = f'https://finance.yahoo.com/quote/{ticker}/options?p={ticker}&date={timestamp_date}'
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

    response = requests.get(url, headers=headers)
    
    prices = pd.concat(
        pd.read_html(response.content)
    ).assign(
        put_call = lambda x: x['Contract Name'].str.extract(r'[A-Z]+\d{6}([C|P])'),
        Bid = lambda x: x.Bid.apply(pd.to_numeric,errors = 'coerce'),
        Ask = lambda x: x.Ask.apply(pd.to_numeric,errors = 'coerce'),
    ).set_index(['put_call','Strike']).dropna()
    
    return prices

def fidelity_option_prices(file_path = '../Option Chain_ Fidelity Investments.html'):
    print('Did you update the html file?')

    with open(file_path,'r', encoding='utf-8', errors='ignore') as fp:
        soup = BeautifulSoup(fp, 'html.parser')
        
    table = soup.find_all('table')
    df = pd.read_html(StringIO(str(table)))[0]

    prices = df.set_index('Strike').filter(regex = '^\d',axis = 0).filter(regex = 'Bid|Ask|Vol')
    prices.index = prices.index.astype(float)
    prices = pd.concat([
        prices.assign(
            put_call = 'C'
        ).set_index('put_call',append=True)[['Bid','Ask','Imp Vol']],
        prices.assign(
            put_call = 'P',
        ).set_index('put_call',append=True)[['Bid.1','Ask.1','Imp Vol.1']].rename(columns = lambda x: x[:-2]),
    ]).reorder_levels(['put_call','Strike']).sort_index().assign(
        IV = lambda x: x['Imp Vol'].apply(lambda y: y[:-2]).replace({'':nan}).astype(float)/100
    ).drop(columns = ['Imp Vol']).astype(float).assign(
        Midpoint = lambda x: rounded_midpoint(x.Bid,x.Ask,0.05)
    )

    return prices

def CBOE_option_prices(filepath,date):

    prices = pd.read_parquet(filepath)

    prices.set_index('strike')

    return prices