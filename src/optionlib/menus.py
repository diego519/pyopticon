import pandas as pd
import numpy as np
from .options import *
from itertools import combinations

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
        menu = dict()
        i = abs(np.asarray(list(strikes)) - self.last_close).argmin()
        ATM = np.asarray(list(strikes))[i]

        # Covered calls
        for i in self.options['calls']['write'].keys():
            opt = self.options['calls']['write'][i]
            menu[('covered call',i,None,None,None)] = [opt.price, opt.payout]

        # Written puts
        for i in self.options['puts']['write'].keys():
            opt = self.options['puts']['write'][i]
            menu[('cash covered put',i,None,None,None)] = [opt.price,opt.payout]
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
            menu[('Bull call spread',i,j,None,None)] = [
                opt.price,opt.payout
            ]
            
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
            menu[('Bear call spread',i,j,None,None)] = [
                opt.price,opt.payout
            ]
        
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
            menu[('Bear put spread',i,j,None,None)] = [
                opt.price,opt.payout
            ]
        
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
            menu[('Bull put spread',i,j,None,None)] = [
                opt.price,opt.payout
            ]
            
        # Long straddles
        combos = (self.options['puts']['buy'].keys() & self.options['calls']['buy'].keys())
        for s in combos:
            opt = OptionChain([
                self.options['puts']['buy'][s],
                self.options['calls']['buy'][s]
            ])
            menu[('Long straddle',s,None,None,None)] = [
                opt.price,opt.payout
            ]
        
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
            menu[('Butterfly spread',l,ATM,ATM,h)] = [opt.price,opt.payout]
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
        print(f'Calculating {len(combos)} Iron Condors...')

        for pl,ph,cl,ch in combos:
            opt = OptionChain([
                self.options['puts']['buy'][pl],
                self.options['puts']['write'][ph],
                self.options['calls']['write'][cl],
                self.options['calls']['buy'][ch],
            ])
            menu[('Iron condor',pl,ph,cl,ch)] = [opt.price, opt.payout]
        print('Iron condors complete')
            
        # Transform output and return                                       
        menu = pd.DataFrame.from_dict(
            menu,
            orient = 'index',
            columns = ['cost','quantiles']
        ).assign(
            EV = lambda x: x.quantiles.apply(lambda y: y.sum()),
            E_pct = lambda x: (x.EV/x.cost).mask(x.cost.lt(0),np.nan),
            win_pct = lambda x: [i.gt(0).mean() for i in x.quantiles],
            E_win = lambda x: [i[i.gt(0)].multiply(100).mean() for i in x.quantiles],
            E_loss = lambda x: [i[i.le(0)].multiply(-100).mean() for i in x.quantiles],
            kelly_criteria = lambda x: x.win_pct - (
                x.win_pct.add(-1).multiply(-1)/(x.E_win/x.E_loss)
            )
        )
        menu.index = pd.MultiIndex.from_tuples(
            menu.index,
            names = ['strategy','leg_1','leg_2','leg_3','leg_4']
        )
        
        return(menu)