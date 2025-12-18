from SmartApi.smartConnect import SmartConnect
import pyotp
import pandas as pd
import requests
import pandas as pd
from datetime import datetime
from datetime import datetime, timedelta
import pandas_ta as ta


api_key = 'C46AImmE'
clientId = 'M507690'
pwd = '4663'
smartApi = SmartConnect(api_key)
token = "4ACNNSIZL6EV6UQ2ACSI23H5VQ"
totp=pyotp.TOTP(token).now()
correlation_id = "abc123"

# login api call

data = smartApi.generateSession(clientId, pwd, totp)
# print(data)
authToken = data['data']['jwtToken']
refreshToken = data['data']['refreshToken']

# fetch the feedtoken
feedToken = smartApi.getfeedToken()

# fetch User Profile
res = smartApi.getProfile(refreshToken)
smartApi.generateToken(refreshToken)
res=res['data']['exchanges']

#fetch User Profile
userProfile= smartApi.getProfile(refreshToken)
User= pd.DataFrame(userProfile)
User = User['data']['name']
User
print(f"Welcome {User} to the World of SMART API BY ANGEL.....!!")
print(f"Login Sucessfull....!!")
