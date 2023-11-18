import numpy as np

class Option():
    def __init__(self,strike, price,last_close,quantiles,call = True,write=False):
        self.strike = strike
        self.price = price * (1 - 2*int(write))
        self.last_close = last_close
        self.quantiles = quantiles
        self.call = call
        self.write = write
        
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
            .multiply(0.01*write_flag)
        )
        
        return(quantile_payout)
    
    def __add__(self,obj1):
        pass

class OptionChain:
    
    def __init__(self, options):
        self.options = options
        self.price = sum([o.price for o in options])
        
        self.payout = sum([o.payout for o in options])
        self.expected_value = self.payout.sum()
        