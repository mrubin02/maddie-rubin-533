import pandas as pd

data = pd.read_csv('data.csv')

columns = ["Trade ID", "Date", "Asset", "Trip", "Action", "Type", "Price", "Status"]
blot_data = []
alpha = -0.01
for i in range(len(data)):
    index = len(data) - 1 - i
    trade_id = i + 1
    asset = "IVV"
    trip = 'ENTER'
    action = 'BUY'
    type = "LMT"
    status = "Submitted"

    price = data["Close"][index] * (1+alpha)
    date = pd.to_datetime(data["Date"][index]) + pd.DateOffset(1)
    row = [trade_id, date, asset, trip, action, type, price, status]
    blot_data += [row]

blotter = pd.DataFrame(columns = columns, data= blot_data)

print(blotter)
cancelled = []
for i in range(len(data) - 3):
    for j in range(1,4):
        rev_ind = len(data) - 1 - i - j
        print(data["Low"][rev_ind])
        print(blotter["Price"][i])
        k = 0
        if (data['Low'][rev_ind]) < blotter["Price"][i]:
            break
        else:
            if (j==3):
                status = "CANCELLED"
                date = data["Date"][rev_ind]
                price = blotter["Price"][i]
                type = "LMT"
                action = "BUY"
                trip = "ENTER"
                asset = "IVV"
                trade_id = blotter["Trade ID"][i]
                row = [trade_id, date, asset, trip, action, type, price, status]
                cancelled += [row]


cancelled_df = pd.DataFrame(columns = columns, data = cancelled)

print(cancelled_df)


