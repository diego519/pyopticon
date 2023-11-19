import pandas as pd

def option_prices(ticker,date):
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
