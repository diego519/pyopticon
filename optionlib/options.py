import numpy as np

class Option():
    def __init__(self,
                 strike, 
                 price,
                 last_close,
                 quantiles,
                 call = True,
                 write=False,
                 commission=0.65):
        
        self.strike = strike
        self.price = price * (1 - 2*int(write))
        self.last_close = last_close
        self.quantiles = quantiles
        self.call = call
        self.write = write
        self.commission = commission
        
        self.payout = self._quantile_payout(call,write)
        self.expected_value = self.payout.sum()
        
    @classmethod
    def write_call(cls,strike,price,last_close,quantiles):
        write_call = Option(strike,price,last_close,quantiles,call=True,write=True)
        return(write_call)
        
    @classmethod
    def buy_call(cls,strike,price,last_close,quantiles):
        buy_call = Option(strike,price,last_close,quantiles,call=True,write=False)
        return(buy_call)
    
    @classmethod
    def write_put(cls,strike,price,last_close,quantiles):
        write_put = Option(strike,price,last_close,quantiles,call=False,write=True)
        return(write_put)
    
    @classmethod
    def buy_put(cls,strike,price,last_close,quantiles):
        buy_put = Option(strike,price,last_close,quantiles,call=False,write=False)
        return(buy_put)
        
    def _quantile_payout(self, call, write):
        
        put_flag = -1 if call == False else 1
        write_flag = -1 if write == True else 1
        
        quantile_payout = (
            self.quantiles
            .add(1)
            .multiply(self.last_close)
            .add(-self.strike)
            .multiply(put_flag)
            .add(-abs(self.price))
            .apply(lambda x: np.clip(x,a_min=-abs(self.price),a_max = None))
            .multiply(write_flag*100)
            .add(-self.commission)
        )
        
        return(quantile_payout)
    
    def __add__(self,obj1):
        pass

class OptionChain:
    
    def __init__(self, options):
        self.options = options
        self.price = sum([o.price for o in options])
        
        self.payout = sum([o.payout for o in options])
        self.expected_value = self.payout.mean()
        self.win_pct = self.payout.gt(0).mean()
        self.max_loss = self.payout.min()
        
        kelly_range = [round(j,2) for j in np.arange(0.1,1,0.05)]
        bankroll = 6e4

        contracts = [
            np.floor(bankroll*k / -min(self.max_loss,-1)) for k in kelly_range
        ]

        EV_harmonic = [
            self.payout.multiply(c).div(bankroll).add(1).cumprod().iloc[-1]**(1/99) for c in contracts
        ]

        self.EV_harmonic = max(EV_harmonic)
        EVh_max_idx = EV_harmonic.index(self.EV_harmonic)
        self.kelly = kelly_range[EVh_max_idx]
