from optionlib.data import ticker, prices, earnings

#################
# Data queries  #
#################

# Ticker
data_ticker = ticker.base_data(risk_ticker='SPY')
print(
    "Ticker data call successful",
    data_ticker.head()
)

# Earnings
data_earnings = earnings.earnings_data('TSLA')
print(
    "Earnings data call successful",
    data_earnings.head()
)

# Option prices
data_prices = prices.option_prices('SPY','2023-09-08')
print(
    "Option price data call successful",
    data_prices.head()
)

#################
# Modeling      #
#################

model_obj = Model(data.fillna(method = 'pad'),10)
model_obj.fit_quantiles()