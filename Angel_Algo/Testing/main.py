import streamlit as st
from streamlit_option_menu import option_menu
import NIFTY_VWAP

st.set_page_config(
    page_title="Stock-Market-Analysis",layout='wide'
)

class MultiApp:

    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        with st.sidebar:
            app = option_menu(
                menu_title='Main-Menu',
                options=['NIFTY_Analysis','PNL Analysis'],
                icons=['house-fill', 'person-circle','bar-chart','bar-chart','bar-chart'],
                menu_icon='chat-text-fill',
                default_index=1,
                styles={
                    "container": {"padding": "5!important", "background-color": 'black'},
                    "icon": {"color": "white", "font-size": "23px"},
                    "nav-link": {"color": "white", "font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "blue"},
                    "nav-link-selected": {"background-color": "#02ab21"},
                }
            )
        
        if app == "NIFTY_Analysis":
            NIFTY_VWAP.app()
            

            

# Create an instance of the MultiApp class
app = MultiApp()

# Run the app
app.run()
