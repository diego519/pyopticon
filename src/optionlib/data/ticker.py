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

    spread_url_csv = 'https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=968&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=T10Y2Y&scale=left&cosd=1976-06-01&coed=2024-02-23&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Daily&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2024-02-23&revision_date=2024-02-23&nd=1976-06-01'

    spread = pd.read_csv(spread_url_csv).rename(
        columns={'DATE':'Date'}
    ).assign(
        Date = lambda x: pd.to_datetime(x.Date)
    ).set_index('Date').replace('.',np.nan).astype(float)

    data = data_price.join(pe).join(spread).join(data_vix).assign(
        shiller_pe = lambda x: x.shiller_pe.fillna(method = 'ffill')
    )

    data.index = pd.to_datetime(data.index)

    data = data.assign(
        jobs_friday = lambda x: ((x.index.day <=7) & (x.index.day_of_week == 4)).astype(int),
        month = lambda x: x.index.month.astype('category')
    )

    return(data)
