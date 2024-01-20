import pandas as pd
from datetime import date
import numpy as np

def base_data(risk_ticker = 'SPY', earnings = False):

    end_ts = int(pd.to_datetime(date.today()+pd.Timedelta(1,unit = 'd')).timestamp())

    data_price_raw = pd.read_csv(f'https://query1.finance.yahoo.com/v7/finance/download/{risk_ticker}?period1=728265600&period2={end_ts}&interval=1d&events=history&includeAdjustedClose=true')

    data_vix_raw = pd.read_csv(f'https://query1.finance.yahoo.com/v7/finance/download/%5EVIX?period1=631238400&period2={end_ts}&interval=1d&events=history&includeAdjustedClose=true')

    windows = [1,2,3,4,5,6,7,30,60,90,180,360]

    data_price = data_price_raw.assign(
        Date = lambda x: pd.to_datetime(x.Date),
        Volume_7d = lambda x: x.Volume.rolling(7).mean()
    ).set_index('Date')

    for t in windows:
        data_price[f'pct_delta_{t}d'] = data_price.Close.pct_change(t)

    data_price = data_price.assign(
        hv30 = lambda x: x.pct_delta_1d.multiply(100).rolling(30).var()
    )

    data_vix = data_vix_raw.assign(
        Date = lambda x: pd.to_datetime(x.Date)
    ).set_index('Date').Close.rename('VIX')

    pe = pd.DataFrame(
        pd.read_html('http://www.multpl.com/shiller-pe/table/by-month')[0]
    ).assign(
        Date = lambda x: pd.to_datetime(x.Date)
    ).set_index('Date')

    pe.columns = ['shiller_pe']

    spread_url_csv = 'https://fred.stlouisfed.org/graph/fredgraph.csv?mode=fred&id=T10Y2Y&vintage_date=2024-01-13&revision_date=2024-01-13&nd=1954-07-01'
    T10Y_url_csv = 'https://fred.stlouisfed.org/graph/fredgraph.csv?mode=fred&id=DGS10&vintage_date=2024-01-13&revision_date=2024-01-13&nd=1954-07-01'

    spread = pd.read_csv(spread_url_csv).rename(
        columns={'DATE':'Date'}
    ).assign(
        Date = lambda x: pd.to_datetime(x.Date)
    ).set_index('Date').replace('.',np.nan).astype(float)

    T10Y = pd.read_csv(T10Y_url_csv).rename(
        columns={'DATE':'Date'}
    ).assign(
        Date = lambda x: pd.to_datetime(x.Date)
    ).set_index('Date').replace('.',np.nan).astype(float)

    data = data_price.join(pe).join(spread).join(T10Y).join(data_vix).assign(
        shiller_pe = lambda x: x.shiller_pe.ffill(),
        DGS10 = lambda x: x.DGS10.ffill()
    )

    data.index = pd.to_datetime(data.index)

    jobs_report = pd.read_excel(
        'https://alfred.stlouisfed.org/release/downloaddates?rid=50&ff=xls',
        skiprows=34,
        usecols=[0]
    ).dropna().rename(
        columns = {'Release Dates:':'Date'}
    ).assign(jobs_friday = 1).set_index('Date')

    data = data.join(jobs_report).assign(
        month = lambda x: x.index.month.astype('category'),
        jobs_friday = lambda x: x.jobs_friday.fillna(0)
    )

    return(data)
