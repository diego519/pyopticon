import pandas as pd
import numpy as np
from .options import *
from itertools import combinations
from joblib import Parallel, delayed
from plotly import express as px
from random import choice, sample

class TradeMenu():
    def __init__(self,
                 input_prices,
                 last_close,
                 quantiles,
                 bankroll = 6e4,
                 bounds = (0,np.inf),
                 midpoint_price = False):
        
        self.prices = input_prices.loc[pd.IndexSlice[:,bounds[0]:bounds[1]+.01],:]
        self.last_close = last_close
        self.prices_bid = self.prices.loc[self.prices['Midpoint' if midpoint_price else 'Bid'].gt(0)]
        self.prices_ask = self.prices.loc[self.prices['Midpoint' if midpoint_price else 'Ask'].gt(0)]
        self.quantiles = quantiles
        self.midpoint_price = midpoint_price
        self.bankroll = bankroll
        self.menu = None
        self.menu_quantiles = None

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

    def _process_menu(self, options, strategy):
        
        menu_dict = {
            (strategy,*(n.strike for n in c.options)):
            [c.price,
             c.expected_value,
             c.win_pct,
             c.max_loss,
             c.EV_harmonic,
             c.kelly]
            for c in options if c is not None
        }
        menu = pd.DataFrame.from_dict(
            menu_dict,
            orient = 'index',
            columns = ['cost','EV_arithmetic','win_pct','max_loss','EV_harmonic','kelly']
        )

        menu.index = pd.MultiIndex.from_tuples(
            menu.index,
            names = ['strategy',*(f'leg_{i+1}' for i in range(len(menu.index[0])-1))]
        )

        if self.menu is not None:
            self.menu = pd.concat([self.menu,menu])
        else:
            self.menu = menu

        # EV_harmonic_upper_bound = self.menu.EV_harmonic.mean() + 6 * self.menu.EV_harmonic.std()

        # self.menu = self.menu.where(lambda x: x.EV_harmonic < EV_harmonic_upper_bound)

        q_dict = {
            (strategy,*(n.strike for n in c.options)):
            c.payout for c in options if c is not None
        }

        quantiles = pd.DataFrame.from_dict(
            q_dict,
            orient = 'index',
        )

        quantiles.index = pd.MultiIndex.from_tuples(
            quantiles.index,
            names = ['strategy',*(f'leg_{i+1}' for i in range(len(quantiles.index[0])-1))]
        )

        if self.menu_quantiles is not None:
            self.menu_quantiles = pd.concat([self.menu_quantiles,quantiles])
        else:
            self.menu_quantiles = quantiles

    def iron_condors_search(self, 
                            max_iterations = 10_000, 
                            tabu_list_size = 100,
                            initial_sample = 1_000,
                            win_pct_skew = 0):
        '''Returns a tuple of the optimal solution strikes along with the last n search targets'''

        combos = [
            (pl,ph,cl,ch) for pl in self.options['puts']['buy'].keys()
            for ph in self.options['puts']['write'].keys() if ph > pl
            for cl in self.options['calls']['write'].keys() if cl >= ph
            for ch in self.options['calls']['buy'].keys() if ch > cl
            and self.options['puts']['buy'][pl].price 
             + self.options['puts']['write'][ph].price
             + self.options['calls']['write'][cl].price 
             + self.options['calls']['buy'][ch].price < 0
            and self.options['puts']['buy'][pl].expected_value 
             + self.options['puts']['write'][ph].expected_value
             + self.options['calls']['write'][cl].expected_value 
             + self.options['calls']['buy'][ch].expected_value > 0
        ]
        self.combos = combos

        dims = [list(set(i[j] for i in combos)) for j in range(len(combos[0]))]
        self.dims = dims

        def _objective_function(idx, win_pct_skew):
            
            opt = OptionChain([
                self.options['puts']['buy'][idx[0]],
                self.options['puts']['write'][idx[1]],
                self.options['calls']['write'][idx[2]],
                self.options['calls']['buy'][idx[3]],
            ])
            return(opt.EV_harmonic*(opt.win_pct**win_pct_skew))

        def _get_neighbors(idx, dims):
            neighbors = list()
            for i in [1,-1]:
                for j in range(len(dims)):
                    if 0 < idx[j] + i < len(dims[j]):
                        neighbor = idx[:]
                        neighbor[j] += i
                        while tuple(neighbor) not in self.combos and 0 < neighbor[j] + i < len(dims[j]):
                            neighbor[j] += i
                        neighbors.append(neighbor)
            return neighbors

        def _tabu_search(initial_solution, max_iterations, tabu_list_size, dims, combos):
            best_solution = initial_solution
            best_solution_strikes = [dims[i][best_solution[i]] for i in range(len(best_solution))]
            best_solution_fitness = _objective_function(best_solution_strikes,win_pct_skew)
            current_solution = initial_solution
            tabu_list = []
        
            for n in range(max_iterations):
                neighbors = _get_neighbors(current_solution,dims)
                best_neighbor = None
                best_neighbor_strikes = None
                best_neighbor_fitness = float(-np.inf)
        
                for neighbor in neighbors:
                    neighbor_strikes = tuple(dims[i][neighbor[i]] for i in range(len(initial_solution)))
                    if neighbor not in tabu_list and neighbor_strikes in combos:
                        neighbor_fitness = _objective_function(neighbor_strikes,win_pct_skew)
                        # print(neighbor, neighbor_fitness)
                        if neighbor_fitness > best_neighbor_fitness:
                            best_neighbor = neighbor
                            best_neighbor_strikes = [dims[i][best_neighbor[i]] for i in range(len(initial_solution))]
                            best_neighbor_fitness = neighbor_fitness
        
                if best_neighbor is None or best_neighbor_fitness < best_solution_fitness:
                    print(f'''Local maximum after {n} iterations''')
                    break
        
                current_solution = best_neighbor
                tabu_list.append(best_neighbor)
                if len(tabu_list) > tabu_list_size:
                    tabu_list.pop(0)

                if _objective_function(best_neighbor_strikes,win_pct_skew) > best_solution_fitness:
                    best_solution = best_neighbor
                    best_solution_strikes = [dims[i][best_solution[i]] for i in range(len(best_solution))]
                    best_solution_fitness = _objective_function(best_solution_strikes,win_pct_skew)

            return best_solution_strikes, tabu_list
        
        sample_strikes = sample(combos,min(initial_sample,len(combos)))

        sample_strikes_EV = Parallel(
            n_jobs = -1, 
            verbose = 1,
            prefer = 'threads'
        )(delayed(_objective_function)(i,win_pct_skew) for i in sample_strikes)


        initial_solution_strikes = sample_strikes[sample_strikes_EV.index(max(sample_strikes_EV))]
        initial_solution = [dims[i].index(initial_solution_strikes[i]) for i in range(len(dims))]

        return _tabu_search(
            initial_solution,
            max_iterations = max_iterations,
            tabu_list_size = tabu_list_size,
            dims = dims,
            combos = combos
        )

    def iron_condors(self,
                     win_pct_bounds = (0.6,0.99),
                     downsample = 100_000):

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
            and self.options['puts']['buy'][pl].expected_value 
             + self.options['puts']['write'][ph].expected_value
             + self.options['calls']['write'][cl].expected_value 
             + self.options['calls']['buy'][ch].expected_value > 0
        ]

        if len(combos) > downsample:
            print(f'{len(combos):,} combinations detected, randomly downsampling to {downsample:,} combinations')
            combos = sample(combos,downsample)

        def iron_condor(pl,ph,cl,ch):
            
            opt = OptionChain([
                self.options['puts']['buy'][pl],
                self.options['puts']['write'][ph],
                self.options['calls']['write'][cl],
                self.options['calls']['buy'][ch],
            ])
            range_bool = (
                (1 < opt.EV_harmonic)
                & (win_pct_bounds[0] <= opt.win_pct <= win_pct_bounds[1])
            )

            if range_bool:
                return opt
        
        print(f'Calculating {len(combos)} Iron Condors...')
        ic_full = Parallel(
            n_jobs = -1, 
            verbose = 1,
            prefer = 'threads'
        )(delayed(iron_condor)(pl,ph,cl,ch) for pl,ph,cl,ch in combos)
        
        self._process_menu(ic_full,'Iron condor')

    def covered_calls(self):
        options = [OptionChain([i]) for i in self.options['calls']['write'].values()]
        self._process_menu(options,'Covered call')

    def naked_put(self):
        options = [OptionChain([i]) for i in self.options['puts']['write'].values()]
        self._process_menu(options,'Naked put')

    def bull_call_spread(self):
        options = [
            OptionChain([
                self.options['calls']['buy'][i],
                self.options['calls']['write'][j]
            ]) for i in self.options['calls']['buy'].keys() 
            for j in self.options['calls']['write'].keys() if i<j
        ]
        self._process_menu(options,'Bull call spread')

    def bear_call_spread(self):
        options = [
            OptionChain([
                self.options['calls']['buy'][i],
                self.options['calls']['write'][j]
            ]) for i in self.options['calls']['buy'].keys() 
            for j in self.options['calls']['write'].keys() if i<j
        ]
        self._process_menu(options,'Bear call spread')
        
    def bear_put_spread(self):
        options = [
            OptionChain([
                self.options['puts']['buy'][i],
                self.options['puts']['write'][j]
            ]) for i in self.options['calls']['buy'].keys() 
            for j in self.options['calls']['write'].keys() if i>j
        ]
        self._process_menu(options,'Bear put spread')
         
    def bull_put_spread(self):
        options = [
            OptionChain([
                self.options['puts']['buy'][i],
                self.options['puts']['write'][j]
            ]) for i in self.options['calls']['buy'].keys() 
            for j in self.options['calls']['write'].keys() if i<j
        ]
        self._process_menu(options,'Bull put spread')

    def spreads(self):
        print('Calculating bull call spreads ...')
        self.bull_call_spread()
        print('Calculating bull put spreads ...')
        self.bull_put_spread()
        print('Calculating bear call spreads ...')
        self.bear_call_spread()
        print('Calculating bear put spreads ...')
        self.bear_put_spread()
        print('Complete')
    
    def long_strangle(self):
        options = [
            OptionChain([
                self.options['puts']['buy'][i],
                self.options['calls']['buy'][j]
            ]) for i in self.options['calls']['buy'].keys() 
            for j in self.options['calls']['write'].keys() if i<=j
        ]
        self._proces_menu(options,'Long strangle/straddle')

    def kelly_criteria(self,
                       strategy,
                       leg_1,
                       leg_2,
                       leg_3,
                       leg_4,
                       bankroll = 6e4,
                       iterations = 1000,
                       time = 50,
                       tabu = True):
        
        if tabu == True:
            menu_slice = OptionChain([
                self.options['puts']['buy'][leg_1],
                self.options['puts']['write'][leg_2],
                self.options['calls']['write'][leg_3],
                self.options['calls']['buy'][leg_4],
            ]).payout
        else:
            menu_slice = self.quantiles.loc[pd.IndexSlice[strategy,leg_1,leg_2,leg_3,leg_4],:].T

        value_at_risk = [round(i,2) for i in np.arange(0.1,1,0.05)]
        max_loss = menu_slice.min()
        max_gain = menu_slice.max()

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
                            * outcome.loc[i,'payouts'],
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