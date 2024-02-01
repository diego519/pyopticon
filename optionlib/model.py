from sklearn.ensemble import GradientBoostingRegressor
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from tqdm import tqdm
from plotly import express as px

class Model():
    '''Object for market prediction model'''
    def __init__(self,
                 base_data,
                 horizon,
                 abs_y = True,
                 earnings = None,
                 earnings_window = False,
                 eval_date = None,
                 jobs_friday_override = False
                ):
        
        self.abs_y = abs_y
        self.earnings = earnings
        self.earnings_window = earnings_window

        self.data = base_data.assign(
            pct_delta_ahead = lambda x: x.Close.pct_change(horizon).shift(-horizon),
            jobs_friday = lambda x: x.jobs_friday.rolling(horizon).max().shift(-horizon)
        )
        self.horizon = horizon
        var_list = '|'.join([
            'pct_delta_\d',
            'shiller'
            'T1',
            'VIX',
            'hv',
            'Volume',
            'jobs_friday',
            'month'
        ])
        
        if eval_date is None:
            self.X = self.data.dropna().filter(regex = var_list)
            self.y = self.data.dropna()[['pct_delta_ahead']]
            self.X_pred = self.data.filter(regex = var_list).tail(1)
        else:
            self.X = self.data.filter(regex = var_list).iloc[
                :self.data.index.get_loc(eval_date) - horizon + 1,:
                ].dropna()
            self.y = self.data.loc[self.X.index,['pct_delta_ahead']]
            self.X_pred = self.data.filter(regex = var_list).loc[[eval_date],:]
        
        horizon_range = pd.bdate_range(self.X_pred.index[0],periods = self.horizon+1)[1:]
        self.X_pred['jobs_friday'] = ((horizon_range.day<=7) & (horizon_range.day_of_week==4)).max().astype(int)

        if jobs_friday_override:
            self.X_pred['jobs_friday'] = 1
        
        if self.earnings is not None:
            self.X = self.X.join(earnings)
            self.X = self.X.assign(
                earnings_window = lambda x: x.earnings_release[::-1].rolling(horizon,min_periods = 1).max().fillna(0)
            ).drop(columns = 'earnings_release')
            
    
    def fit_quantiles(self,cores = -1):

        if self.earnings is not None:
            self.X_pred['earnings_window'] = int(self.earnings_window)

        def qtile_fit(q):
            mod = GradientBoostingRegressor(loss = 'quantile',alpha = q/100)
            if self.abs_y:
                mod.fit(self.X,self.y.abs().values.ravel())
            else:
                mod.fit(self.X,self.y.values.ravel())
            return mod.predict(self.X_pred)[0]
        
        qtile_out = Parallel(n_jobs=cores,verbose = 10)(delayed(qtile_fit)(i) for i in range(1,100))
        
        qtile_model = pd.DataFrame(
            qtile_out,
            index = [i/100 for i in range(1,100)],
            columns = ['model_qtiles']
        )
        
        if self.abs_y:
            qtile_obs_abs = self.y.abs().rename(columns = {'pct_delta_ahead':'observed_abs'}).quantile(
                [round(i,2) for i in np.linspace(0.01,.99,99)]
            )

            qtile_merge = qtile_model.join(qtile_obs_abs,how = 'outer')

            self.model_qtiles = qtile_merge

            adj_factor = qtile_merge.sum().model_qtiles/qtile_merge.sum().observed_abs

            qtile_obs = self.y.rename(columns = {'pct_delta_ahead':'observed'}).quantile(
                [round(i,2) for i in np.linspace(0.01,.99,99)]
            )

            quantiles = qtile_obs.assign(
                reweighted_tails = lambda x: x.observed*adj_factor
            )
            
            self.quantiles = quantiles[['observed','reweighted_tails']]
            self.adj_factor = adj_factor
            print('Adjustment factor:',round(adj_factor,4))
        else:
            quantiles = self.y.rename(columns = {'pct_delta_ahead':'observed'}).quantile(
                [round(i,2) for i in np.linspace(0.01,.99,99)]
            ).join(qtile_model).rename(columns = {'model_qtiles':'reweighted'})
            
            self.quantiles = quantiles[['observed','reweighted',]]

    def VIX_diagnostic(self,
                       adj_factor_history,
                       cutoff_date):
         
        adj_factor_df = adj_factor_history.join(
            self.y.abs().rename(columns={'pct_delta_ahead':'pct_delta_ahead_abs'})
        ).join(
            self.X.VIX
        ).join(self.y).loc[cutoff_date:].assign(
            log_delta = lambda x: np.log(x.pct_delta_ahead_abs*100)
        )

        print('Correlation matrix:',adj_factor_df.corr())

        px.scatter(
            adj_factor_df,
            x = 'VIX',
            y = 'pct_delta_ahead_abs',
            title = f'VIX vs. absolute returns over {self.horizon} days',
            hover_name = adj_factor_df.index
        ).show()

        px.scatter(
            adj_factor_df,
            x = 'quantile_backtest',
            y = 'pct_delta_ahead_abs',
            title = f'Adjustment factor vs. absolute returns over {self.horizon} days',
            hover_name = adj_factor_df.index
        ).show()

        px.scatter(
            adj_factor_df,
            x = 'quantile_backtest',
            y = 'VIX',
            color = 'log_delta',
            hover_name = adj_factor_df.index
        )


def fit_historical_quantiles(horizon, data, cores = -1,path = None, dt_range = None):
    if path is None:
        path = f'horizon_{horizon}_adj_factor.csv'

    if dt_range is None:
        dt_range = data.X.index

    output = dict()

    for dt in tqdm(dt_range):
        model_obj = Model(
            data.ffill(),
            horizon,
            eval_date=dt
        )
        model_obj.fit_quantiles(cores = cores)
        output[dt] = model_obj.adj_factor.copy()

    output = pd.Series(output,name = f'adj_factor_{horizon}')
    output.to_csv(path)