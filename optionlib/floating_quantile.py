import pandas as pd
import numpy as np

def floating_quantile(iv, y, iv_hist, ntile = 5, n_years = 10):
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
    iv_hist_var = iv_hist.columns[0]
    iv_hist = iv_hist.join(y,how = 'inner').iloc[-252*n_years:]

    caliper = round(iv_hist.shape[0]/(ntile/2))

    qtile_df_low = iv_hist.sort_values(iv_hist_var).where(
        lambda x: x[iv_hist_var].lt(iv)
    ).dropna().tail(caliper)

    qtile_df_hi = iv_hist.sort_values(iv_hist_var).where(
        lambda x: x[iv_hist_var].gt(iv)
    ).dropna().head(caliper)

    floating_quintile = pd.concat(
        [qtile_df_low,qtile_df_hi]
    ).pct_delta_ahead.quantile([round(i,2) for i in np.linspace(0.01,.99,99)])

    return floating_quintile