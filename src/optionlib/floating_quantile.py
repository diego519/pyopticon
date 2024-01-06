import pandas as pd
import numpy as np

def floating_quantile(adj_factor, y, adj_factor_hist, ntile = 5, n_years = 10):
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
    adj_factor_hist.columns = ['Date','quantile_backtest']
    adj_factor_hist = adj_factor_hist.assign(Date = lambda x: pd.to_datetime(x.Date)).set_index('Date')

    adj_factor_hist = adj_factor_hist.join(y).iloc[-252*n_years:]
    adj_factor_hist = adj_factor_hist.assign(
        adj_factor_decile = lambda x: x.quantile_backtest
            .rank(pct = True).multiply(5).apply(np.floor).add(1).astype(int).replace({6:5})
    )

    caliper = round(adj_factor_hist.shape[0]/(ntile/2))

    qtile_df_low = adj_factor_hist.sort_values('quantile_backtest').where(
        lambda x: x.quantile_backtest.lt(adj_factor)
    ).dropna().tail(caliper)

    qtile_df_hi = adj_factor_hist.sort_values('quantile_backtest').where(
        lambda x: x.quantile_backtest.gt(adj_factor)
    ).dropna().head(caliper)

    floating_quintile = pd.concat(
        [qtile_df_low,qtile_df_hi]
    ).pct_delta_ahead.quantile([round(i,2) for i in np.linspace(0.01,.99,99)])

    return floating_quintile