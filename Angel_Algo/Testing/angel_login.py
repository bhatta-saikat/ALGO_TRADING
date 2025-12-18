from SmartApi.smartConnect import SmartConnect
import pyotp
import pandas as pd
import pyotp

api_key = 'C46AImmE'
clientId = 'M507690'
pwd = '4663'
smartApi = SmartConnect(api_key)
token = "4ACNNSIZL6EV6UQ2ACSI23H5VQ"
totp=pyotp.TOTP(token).now()
correlation_id = "abc123"

def angel_login(api_key, client_id, password, totp_secret):
    obj = SmartConnect(api_key=api_key)
    data = obj.generateSession(
        client_id,
        password,
        pyotp.TOTP(totp_secret).now()
    )
    return obj


