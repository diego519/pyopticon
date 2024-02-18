import pandas as pd
import numpy as np
import datetime

def floating_quantile(dte,
                      iv, 
                      y, 
                      iv_hist,
                      time = datetime.time(10,30),
                      ntile = 5, 
                      n_years = 10):
    '''
    Floating quantile of expected returns around a given adjustment factor value.

    Parameters
    ----------
    adj_factor : The current adjustment factor given current market conditions
    y : The pandas Series of returns
    adj_factor_hist : The pandas Series of historical adjustment factors, aligning to the y parameter
    ntile : The quantile size to use for the floating quantile, set to 5 for quintiles
    n_years : The past number of years to use for the floating quantile
    '''
    iv_hist_var = f'IV_{dte}'
    df = iv_hist.groupby('quote_datetime').mean()[[iv_hist_var]]

    dates = pd.DataFrame(
        {'Date':y.Date.unique()}
    ).assign(
        expiration_ts = lambda x: pd.to_datetime(x.Date.shift(-dte)) + pd.to_timedelta(16.25,'h')
    ).set_index('Date')

    prices_open = y[['open']].join(dates).dropna().set_index('expiration_ts',append = True)
    prices_close = y.reset_index(drop=True).rename(
        columns={'quote_datetime':'expiration_ts'}
    ).set_index('expiration_ts')[['close']]

    prices_join = prices_open.join(prices_close).reset_index('quote_datetime',drop=False)

    prices_join = prices_join.where(
        lambda x: x.quote_datetime.dt.time == time
    ).dropna().set_index('quote_datetime',append = True).assign(
        pct_delta_ahead = lambda x: x.close / x.open -1
    )

    df = df.dropna().join(prices_join[['pct_delta_ahead']],how = 'inner')
    df = df.iloc[-252*n_years:]

    caliper = round(df.shape[0]/(ntile/2))

    qtile_df_low = df.sort_values(iv_hist_var).where(
        lambda x: x[iv_hist_var].lt(iv)
    ).dropna().tail(caliper)

    qtile_df_hi = df.sort_values(iv_hist_var).where(
        lambda x: x[iv_hist_var].gt(iv)
    ).dropna().head(caliper)

    floating_quintile = pd.concat(
        [qtile_df_low,qtile_df_hi]
    ).pct_delta_ahead.quantile([round(i,2) for i in np.linspace(0.01,.99,99)])

    return floating_quintile