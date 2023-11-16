import pandas as pd

def earnings_data(ticker):
    url = f'https://finance.yahoo.com/calendar/earnings?symbol={ticker}'
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

    response = requests.get(url, headers=headers)
    
    earnings = pd.read_html(response.content)[0].assign(
        earnings_str = lambda x: x['Earnings Date'].str.extract('(.+),'),
        Date = lambda x: pd.to_datetime(x.earnings_str),
        earnings_release = 1
    ).set_index('Date')[['earnings_release']].sort_index()
    
    return earnings
