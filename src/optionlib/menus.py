import pandas as pd
import numpy as np
from .options import *
from itertools import combinations
from joblib import Parallel, delayed

class TradeMenu():
    def __init__(self,input_prices,last_close,quantiles,bounds = (0,np.inf)):
        self.prices = input_prices.loc[pd.IndexSlice[:,bounds[0]:bounds[1]],:]
        self.last_close = last_close
        self.prices_bid = self.prices.loc[self.prices.Bid.gt(0)]
        self.prices_ask = self.prices.loc[self.prices.Ask.gt(0)]
        self.quantiles = quantiles
        self.options = self._create_options()
        self.menu = self._run_menu()

    def _create_options(self):
        options = {
            o:{j:dict() for j in ['write','buy']}
            for o in ['calls','puts']
        }
        
        for o,s in self.prices_bid.index:
            if o == 'P':
                options['puts']['write'][s] = Option.write_put(
                    s,self.prices_bid.loc[(o,s),'Bid'],self.last_close,self.quantiles)
            elif o == 'C':
                options['calls']['write'][s] = Option.write_call(
                    s,self.prices_bid.loc[(o,s),'Bid'],self.last_close,self.quantiles)
        
        for o,s in self.prices_ask.index:
            if o == 'P':
                options['puts']['buy'][s] = Option.buy_put(
                    s,self.prices_ask.loc[(o,s),'Ask'],self.last_close,self.quantiles)
            elif o == 'C':
                options['calls']['buy'][s] = Option.buy_call(
                    s,self.prices_ask.loc[(o,s),'Ask'],self.last_close,self.quantiles)
                
        return(options)

    def _run_menu(self):
        
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
        print(f'Calculating {len(combos)} Iron Condors...')
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
            and self.options['puts']['write'][pl].price 
             + self.options['puts']['buy'][ph].price
             + self.options['calls']['buy'][cl].price 
             + self.options['calls']['write'][ch].price < 0
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

        ic_full = Parallel(
            n_jobs = -1, 
            verbose = 3,
            prefer = 'threads'
        )(delayed(iron_condor)(pl,ph,cl,ch) for pl,ph,cl,ch in combos)
        
        for i in ic_full:
            key = list(i.keys())[0]
            menu_dict[key] = i[key][0]
            payout_dict[key] = i[key][1]

        ric_full = Parallel(
            n_jobs = -1, 
            verbose = 3,
            prefer = 'threads'
        )(delayed(reverse_iron_condor)(pl,ph,cl,ch) for pl,ph,cl,ch in reverse_combos)

        for i in ric_full:
            key = list(i.keys())[0]
            menu_dict[key] = i[key][0]
            payout_dict[key] = i[key][1]
            
        print('Iron condors complete')
            
        # Transform output and return       
        self.quantiles = pd.DataFrame.from_dict(
            payout_dict,
            orient = 'index'
        )

        self.menu = pd.DataFrame.from_dict(
            menu_dict,
            orient = 'index',
            columns = ['cost']
        ).assign(
            EV = self.quantiles.sum(1),
            E_pct = lambda x: (x.EV/x.cost).mask(x.cost.lt(0),np.nan),
            win_pct = self.quantiles.gt(0).mean(1),
            E_win = self.quantiles.where(self.quantiles.gt(0)).mean(1).multiply(100),
            E_loss = self.quantiles.where(self.quantiles.lt(0)).mean(1).multiply(100),
            max_loss = self.quantiles.min(1),
            kelly_criteria_E_loss = lambda x: x.win_pct - (
                x.win_pct.add(-1).multiply(-1)/(x.E_win/x.E_loss)
            ),
            kelly_criteria_max_loss = lambda x: x.win_pct - (
                x.win_pct.add(-1).multiply(-1)/(x.E_win/x.max_loss)
            )
        )
        self.menu.index = pd.MultiIndex.from_tuples(
            self.menu.index,
            names = ['strategy','leg_1','leg_2','leg_3','leg_4']
        )