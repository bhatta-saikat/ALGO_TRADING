# BhavCopy.py
from nsepy import get_history
import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import pandas_ta as ta
from datetime import date
from nsepy import get_history
import datetime
import numpy as np
import yfinance as yf
import mplfinance as mpf

import mysql.connector
from nsedt import equity as eq
from tradingview_ta import TA_Handler, Interval, Exchange
import tradingview_ta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date
import pandas as pd
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import streamlit as st
import streamlit as st
from streamlit_option_menu import option_menu

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import datetime
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
#import Fyers
import warnings
warnings.filterwarnings('ignore')
import time
import os
import datetime
import pandas as pd
import json
import requests
import time
import pyotp
import os
import requests
from urllib.parse import parse_qs,urlparse
import sys
import warnings
warnings.filterwarnings('ignore')
import pandas_ta as ta
from fyers_api import fyersModel, accessToken
from datetime import datetime, timedelta
import time
import os
import datetime
import pandas as pd
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt  # Import Matplotlib
#import Angel

from nsepy import get_history
from datetime import date
import pandas as pd
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from nselib import derivatives
import numpy as np
npNaN = np.nan

import mysql.connector
import pandas as pd
import requests
import numpy as np
import numpy as np
import pandas as pd
import yfinance as yf
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
from ta import add_all_ta_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import plotly.graph_objects as go


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

import mysql.connector
import pandas as pd
import requests
import numpy as np
import numpy as np
import pandas as pd
import yfinance as yf
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
from ta import add_all_ta_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import plotly.graph_objects as go


def app():
    st.subheader("NIFTY50 Price Prediction By Data Analysis :")
    selected = option_menu(
        menu_title=None,  # required
        options=["VWAP-Strategy","", "VWMA-Strategy", "EMA-SMA Strategy"],  # required
        icons=["house", "book", "envelope"],  # optional
        menu_icon="cast",  # optional
        default_index=0,  # optional
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#818589"},
            "icon": {"color": "orange", "font-size": "20px"},
            "nav-link": {
                "font-size": "20px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {"background-color": "blue"},
        },
    )


    if selected == "VWAP-Strategy":
        import pyotp
        import pandas as pd
        import plotly.graph_objects as go
        from datetime import datetime, timedelta
        from angel_login import angel_login
        from indicators import add_indicators
        import Angel

        # ---------- PAGE CONFIG ---------- #
        st.set_page_config(layout="wide", page_title="Angel NIFTY Live Dashboard")

        st.title("📈 NIFTY 5-Min Live Chart (Angel One)")

        

        # ---------- FETCH 5-MIN CANDLES ---------- #
        to_date = datetime.now()
        from_date = to_date - timedelta(days=5)

        params = {
            "exchange": "NSE",
            "symboltoken": "99926000",
            "interval": "FIVE_MINUTE",
            "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
            "todate": to_date.strftime("%Y-%m-%d %H:%M")
        }

        data = Angel.smartApi.getCandleData(params)
        df = pd.DataFrame(
            data["data"],
            columns=["Datetime", "Open", "High", "Low", "Close", "Volume"]
        )
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df = add_indicators(df)
        df = df.tail(100)

        # ---------- CANDLESTICK + VWAP + EMA ---------- #
        fig = go.Figure()

        fig.add_candlestick(
            x=df["Datetime"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="NIFTY"
        )

        fig.add_trace(go.Scatter(
            x=df["Datetime"],
            y=df["VWAP"],
            line=dict(width=2),
            name="VWAP"
        ))

        fig.add_trace(go.Scatter(
            x=df["Datetime"],
            y=df["EMA20"],
            line=dict(width=1),
            name="EMA 20"
        ))

        fig.update_layout(
            height=500,
            xaxis_rangeslider_visible=False,
            title="NIFTY 5-Min Candles with VWAP & EMA"
        )

        st.plotly_chart(fig, use_container_width=True)

        # ---------- RSI ---------- #
        st.subheader("RSI")
        st.line_chart(df.set_index("Datetime")["RSI"])

        # ---------- MACD ---------- #
        st.subheader("MACD")
        st.line_chart(df.set_index("Datetime")[["MACD", "MACD_SIGNAL"]])

        # ---------- AUTO REFRESH ---------- #
        from streamlit_autorefresh import st_autorefresh

        st.caption("🔄 Auto refresh every 10 seconds")

        st_autorefresh(
            interval=30 * 1000,  # 60 seconds
            key="nifty_refresh"
        )






