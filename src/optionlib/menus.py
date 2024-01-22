import pandas as pd
import numpy as np
from .options import *
from itertools import combinations
from joblib import Parallel, delayed
from plotly import express as px

class TradeMenu():
    def __init__(self,
                 input_prices,
                 last_close,
                 quantiles,
                 bankroll = 6e4,
                 bounds = (0,np.inf),
                 midpoint_price = False):
        
        self.prices = input_prices.loc[pd.IndexSlice[:,bounds[0]:bounds[1]],:]
        self.last_close = last_close
        self.prices_bid = self.prices.loc[self.prices['Midpoint' if midpoint_price else 'Bid'].gt(0)]
        self.prices_ask = self.prices.loc[self.prices['Midpoint' if midpoint_price else 'Bid'].gt(0)]
        self.quantiles = quantiles
        self.midpoint_price = midpoint_price
        self.bankroll = bankroll

        options = {
            o:{j:dict() for j in ['write','buy']}
            for o in ['calls','puts']
        }

        for o,s in self.prices_bid.index:
            price = 'Midpoint' if self.midpoint_price else 'Bid'
            if o == 'P':
                options['puts']['write'][s] = Option.write_put(
                    s,self.prices_bid.loc[(o,s),price],self.last_close,self.quantiles)
            elif o == 'C':
                options['calls']['write'][s] = Option.write_call(
                    s,self.prices_bid.loc[(o,s),price],self.last_close,self.quantiles)
        
        for o,s in self.prices_ask.index:
            price = 'Midpoint' if self.midpoint_price else 'Ask'
            if o == 'P':
                options['puts']['buy'][s] = Option.buy_put(
                    s,self.prices_ask.loc[(o,s),price],self.last_close,self.quantiles)
            elif o == 'C':
                options['calls']['buy'][s] = Option.buy_call(
                    s,self.prices_ask.loc[(o,s),price],self.last_close,self.quantiles)
                
        self.options = options

    def iron_condors(self):
        menu_dict = dict()
        payout_dict = dict()

        # Iron condors/butterflies
        
        combos = [
            (pl,ph,cl,ch) for pl in self.options['puts']['buy'].keys()
            for ph in self.options['puts']['write'].keys() if ph > pl
            for cl in self.options['calls']['write'].keys() if cl >= ph
            for ch in self.options['calls']['buy'].keys() if ch > cl
            and self.options['puts']['buy'][pl].price 
             + self.options['puts']['write'][ph].price
             + self.options['calls']['write'][cl].price 
             + self.options['calls']['buy'][ch].price < 0
        ]

        def iron_condor(pl,ph,cl,ch):
            
            ic_dict = dict()
            
            opt = OptionChain([
                self.options['puts']['buy'][pl],
                self.options['puts']['write'][ph],
                self.options['calls']['write'][cl],
                self.options['calls']['buy'][ch],
            ])
            ic_dict[('Iron condor',pl,ph,cl,ch)] = [opt.price, opt.payout]

            return ic_dict
        
        print(f'Calculating {len(combos)} Iron Condors...')
        ic_full = Parallel(
            n_jobs = -1, 
            verbose = 1,
            prefer = 'threads'
        )(delayed(iron_condor)(pl,ph,cl,ch) for pl,ph,cl,ch in combos)
        
        for i in ic_full:
            key = list(i.keys())[0]
            menu_dict[key] = i[key][0]
            payout_dict[key] = i[key][1]
            
        print('Iron condors complete. Transforming payout quantiles...')
            
        # Transform output and return       
        self.quantiles = pd.DataFrame.from_dict(
            payout_dict,
            orient = 'index'
        )

        kelly_range = [round(j,2) for j in np.arange(0.1,1,0.05)]

        self.EV_harmonic = pd.DataFrame(
            index = self.quantiles.index,
            columns = kelly_range
        )

        for k in kelly_range:
            contracts = self.quantiles.min(1).multiply(1e4).apply(
                lambda x: np.floor(self.bankroll*k/-min(x,-1))
            )

            self.EV_harmonic.loc[:,k] = \
                self.quantiles.multiply(contracts*1e4,axis = 0)\
                .div(self.bankroll)\
                .add(1)\
                .cumprod(axis = 1)\
                .iloc[:,-1]**(1/99)

        print('Calculating payout characteristics')
        self.menu = pd.DataFrame.from_dict(
            menu_dict,
            orient = 'index',
            columns = ['cost']
        ).assign(
            EV_arithmetic = self.quantiles.sum(1),
            E_pct = lambda x: (x.EV_arithmetic/x.cost).mask(x.cost.lt(0),np.nan),
            win_pct = self.quantiles.gt(0).mean(1),
            E_win = self.quantiles.where(self.quantiles.gt(0)).mean(1).multiply(100),
            E_loss = self.quantiles.where(self.quantiles.lt(0)).mean(1).multiply(100),
            max_loss = self.quantiles.min(1).multiply(100),
            EV_harmonic = self.EV_harmonic.max(axis = 1),
            kelly_criteria_EV_harmonic = self.EV_harmonic.idxmax(axis = 1)
        )
        self.menu.index = pd.MultiIndex.from_tuples(
            self.menu.index,
            names = ['strategy','leg_1','leg_2','leg_3','leg_4']
        )

    def run_menu(self):
        
        strikes = self.prices.index.get_level_values('Strike').unique()
        menu_dict = dict()
        payout_dict = dict()
        i = abs(np.asarray(list(strikes)) - self.last_close).argmin()
        ATM = np.asarray(list(strikes))[i]

        # Covered calls
        for i in self.options['calls']['write'].keys():
            opt = self.options['calls']['write'][i]
            menu_dict[('covered call',i,None,None,None)] = opt.price
            payout_dict[('covered call',i,None,None,None)] = opt.payout

        # Written puts
        for i in self.options['puts']['write'].keys():
            opt = self.options['puts']['write'][i]
            menu_dict[('cash covered put',i,None,None,None)] = opt.price
            payout_dict[('cash covered put',i,None,None,None)] = opt.payout
        print('Write strategies complete')
            
        # Bull call spreads
        combos = [
            (i,j) for i in self.options['calls']['buy'].keys() 
            for j in self.options['calls']['write'].keys() if i<j
        ]
        
        for i,j in combos:
            opt = OptionChain([
                self.options['calls']['buy'][i],
                self.options['calls']['write'][j]
            ])
            menu_dict[('Bull call spread',i,j,None,None)] = opt.price
            payout_dict[('Bull call spread',i,j,None,None)] = opt.payout
            
            
        # Bear call spreads
        combos = [
            (i,j) for i in self.options['calls']['buy'].keys() 
            for j in self.options['calls']['write'].keys() if i>j
        ]
        
        for i,j in combos:
            opt = OptionChain([
                self.options['calls']['buy'][i],
                self.options['calls']['write'][j]
            ])
            menu_dict[('Bear call spread',i,j,None,None)] = opt.price
            payout_dict[('Bear call spread',i,j,None,None)] = opt.payout
        
        # Bear put spreads
        combos = [
            (i,j) for i in self.options['puts']['buy'].keys() 
            for j in self.options['puts']['write'].keys() if i>j
        ]
        
        for i,j in combos:
            opt = OptionChain([
                self.options['puts']['buy'][i],
                self.options['puts']['write'][j]
            ])
            menu_dict[('Bear put spread',i,j,None,None)] = opt.price
            payout_dict[('Bear put spread',i,j,None,None)] = opt.payout
        
        # Bull put spreads
        combos = [
            (i,j) for i in self.options['puts']['buy'].keys() 
            for j in self.options['puts']['write'].keys() if i<j
        ]
        
        for i,j in combos:
            opt = OptionChain([
                self.options['puts']['buy'][i],
                self.options['puts']['write'][j]
            ])
            menu_dict[('Bull put spread',i,j,None,None)] = opt.price
            payout_dict[('Bull put spread',i,j,None,None)] = opt.payout
            
        # Long straddles
        combos = (self.options['puts']['buy'].keys() & self.options['calls']['buy'].keys())
        for s in combos:
            opt = OptionChain([
                self.options['puts']['buy'][s],
                self.options['calls']['buy'][s]
            ])
            menu_dict[('Long straddle',s,None,None,None)] = opt.price
            payout_dict[('Long straddle',s,None,None,None)] = opt.payout
        
        # Long strangle
        # To be implemented...
        
        # Butterfly spreads
        combos = [
            (i,j) for i,j in combinations(self.options['calls']['buy'].keys(),2) 
            if i<ATM and j>ATM
        ]
        
        for l,h in combos:
            opt = OptionChain([
                self.options['calls']['buy'][l],
                self.options['calls']['write'][ATM],
                self.options['calls']['write'][ATM],
                self.options['calls']['buy'][h]
            ])

            menu_dict[('Butterfly spread',l,ATM,ATM,h)] = opt.price
            payout_dict[('Butterfly spread',l,ATM,ATM,h)] = opt.payout
        print('Spreads and straddles complete')
            
        # Iron condors/butterflies
        
        combos = [
            (pl,ph,cl,ch) for pl in self.options['puts']['buy'].keys()
            for ph in self.options['puts']['write'].keys() if ph > pl
            for cl in self.options['calls']['write'].keys() if cl >= ph
            for ch in self.options['calls']['buy'].keys() if ch > cl
            and self.options['puts']['buy'][pl].price 
             + self.options['puts']['write'][ph].price
             + self.options['calls']['write'][cl].price 
             + self.options['calls']['buy'][ch].price < 0
        ]

        reverse_combos = [
            (pl,ph,cl,ch) for pl in self.options['puts']['write'].keys()
            for ph in self.options['puts']['buy'].keys() if ph > pl
            for cl in self.options['calls']['buy'].keys() if cl >= ph
            for ch in self.options['calls']['write'].keys() if ch > cl
        ]

        def iron_condor(pl,ph,cl,ch):
            
            ic_dict = dict()
            
            opt = OptionChain([
                self.options['puts']['buy'][pl],
                self.options['puts']['write'][ph],
                self.options['calls']['write'][cl],
                self.options['calls']['buy'][ch],
            ])
            ic_dict[('Iron condor',pl,ph,cl,ch)] = [opt.price, opt.payout]

            return ic_dict
        
        def reverse_iron_condor(pl,ph,cl,ch):
            
            ic_dict = dict()

            opt = OptionChain([
                self.options['puts']['write'][pl],
                self.options['puts']['buy'][ph],
                self.options['calls']['buy'][cl],
                self.options['calls']['write'][ch],
            ])
            ic_dict[('Reverse iron condor',pl,ph,cl,ch)] = [opt.price, opt.payout]
            
            return ic_dict
        
        print(f'Calculating {len(combos)} Iron Condors...')
        ic_full = Parallel(
            n_jobs = -1, 
            verbose = 1,
            prefer = 'threads'
        )(delayed(iron_condor)(pl,ph,cl,ch) for pl,ph,cl,ch in combos)
        
        for i in ic_full:
            key = list(i.keys())[0]
            menu_dict[key] = i[key][0]
            payout_dict[key] = i[key][1]

        print(f'Calculating {len(reverse_combos)} Reverse Iron Condors...')
        ric_full = Parallel(
            n_jobs = -1, 
            verbose = 1,
            prefer = 'threads'
        )(delayed(reverse_iron_condor)(pl,ph,cl,ch) for pl,ph,cl,ch in reverse_combos)

        for i in ric_full:
            key = list(i.keys())[0]
            menu_dict[key] = i[key][0]
            payout_dict[key] = i[key][1]
            
        print('Iron condors complete. Transforming payout quantiles...')
            
        # Transform output and return       
        self.quantiles = pd.DataFrame.from_dict(
            payout_dict,
            orient = 'index'
        )

        kelly_range = [round(j,2) for j in np.arange(0.1,1,0.05)]

        self.EV_harmonic = pd.DataFrame(
            index = self.quantiles.index,
            columns = kelly_range
        )

        for k in kelly_range:
            contracts = self.quantiles.min(1).multiply(1e4).apply(
                lambda x: np.floor(self.bankroll*k/-min(x,-1))
            )

            self.EV_harmonic.loc[:,k] = \
                self.quantiles.multiply(contracts*1e4,axis = 0)\
                .div(self.bankroll)\
                .add(1)\
                .cumprod(axis = 1)\
                .iloc[:,-1]**(1/99)

        print('Calculating payout characteristics')
        self.menu = pd.DataFrame.from_dict(
            menu_dict,
            orient = 'index',
            columns = ['cost']
        ).assign(
            EV_arithmetic = self.quantiles.sum(1),
            E_pct = lambda x: (x.EV_arithmetic/x.cost).mask(x.cost.lt(0),np.nan),
            win_pct = self.quantiles.gt(0).mean(1),
            E_win = self.quantiles.where(self.quantiles.gt(0)).mean(1).multiply(100),
            E_loss = self.quantiles.where(self.quantiles.lt(0)).mean(1).multiply(100),
            max_loss = self.quantiles.min(1).multiply(100),
            EV_harmonic = self.EV_harmonic.max(axis = 1),
            kelly_criteria_EV_harmonic = self.EV_harmonic.idxmax(axis = 1)
        )
        self.menu.index = pd.MultiIndex.from_tuples(
            self.menu.index,
            names = ['strategy','leg_1','leg_2','leg_3','leg_4']
        )

    def kelly_criteria(self,
                       strategy,
                       leg_1,
                       leg_2,
                       leg_3,
                       leg_4,
                       bankroll = 6e4,
                       iterations = 1000,
                       time = 50):
        
        menu_slice = self.quantiles.loc[pd.IndexSlice[strategy,leg_1,leg_2,leg_3,leg_4],:].T

        value_at_risk = [round(i,2) for i in np.arange(0.1,1,0.05)]
        max_loss = menu_slice.min()*1e4
        max_gain = menu_slice.max()*1e4

        def kelly_sim(n):
            outcome = pd.DataFrame(index = range(time+1),columns = value_at_risk)
            outcome.loc[:,'payouts'] = menu_slice.sample(time+1,replace = True).values
            outcome.loc[0,:] = bankroll

            for j in value_at_risk:
                for i in outcome.index[1:]:
                    if outcome.loc[i-1,j]*j/(-max_loss) < 0:
                        outcome.loc[i,j] = 0
                    else:
                        outcome.loc[i,j] = np.max([
                            outcome.loc[i-1,j] 
                            + round(outcome.loc[i-1,j]*j/(-max_loss),0)
                            * outcome.loc[i,'payouts']*10000,
                            0
                        ])
                        
            return (n,outcome.loc[time,:].drop(columns = 'payouts'))

        sims = Parallel(n_jobs=-1,verbose = 5)(delayed(kelly_sim)(i) for i in range(iterations))

        kc_output = pd.DataFrame(index = range(iterations),columns = value_at_risk)
        for i,j in sims:
            kc_output.loc[i,:] = j


        x_var_name = f'Median outcome at t={time} with {iterations} iterations'
        y_var_name = 'Probability of loss'
        kelly_curve = pd.DataFrame().from_dict({
            x_var_name:kc_output.median(),
            y_var_name:kc_output.lt(bankroll).mean()
        })

        px.line(
            kelly_curve,
            x = x_var_name,
            y = y_var_name,
            hover_name=kc_output.columns,
            markers=True
        ).show()

        px.box(
            kc_output,
            log_y = True,
            height = 600,
            title = 'Distribution of ending values'
        ).show()

        print(pd.DataFrame().from_dict({'Kelly':value_at_risk}).assign(
            Contracts = lambda x: np.floor(bankroll*x.Kelly/(-max_loss)),
            Max_loss = lambda x: max_loss*x.Contracts,
            Max_gain = lambda x: round(max_gain*x.Contracts,2)
        ).set_index('Kelly'))