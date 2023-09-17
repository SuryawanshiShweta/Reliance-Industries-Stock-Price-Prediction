import pandas as pd
import streamlit as st
import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import pickle
import plotly.graph_objects as go
import plotly as px
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

# Load the KMeans model from the pickle file
filename = 'LSTM_model_f.pkl'
loaded_LSTM = pickle.load(open(filename, 'rb'))

data = pd.read_csv('Reliance_dataset (2000 -2022 ).csv')
data['Date']=pd.to_datetime(data['Date'], format='%Y-%m-%d')

#Page input from user
st.sidebar.header('Choose an Info Type')
infoType = st.sidebar.radio(
    "",('Technical','Fundamental')
    )

if(infoType =='Technical'):
    pass
else:
    pass

if(infoType=='Technical'):
     # Set Streamlit config option to disable the warning
    st.set_option('deprecation.showPyplotGlobalUse', False)

# convert an array of values into a dataset matrix
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)


    def main():
    # Set Streamlit app title
     st.title('Reliance Industries Ltd')    
    
    df=pd.read_csv('Reliance_dataset (2000 -2022 ).csv')
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    #st.write(df)
    
    st.subheader('Reliance share close price plot (2000-2021)')
    
    period=('Last 7 days','Last 1 Month','Last 6 Months','Last year')
    button = st.selectbox('Choose Period for Visualization',period)
     
    if button == 'Complete':
        fig= px.line(df,x='Date', y='Close')
        st.plotly_chart(fig)
      
    elif button=='Last 7 days':
        end_date= df['Date'].max()
        start_date= end_date - timedelta(7)
        _7days_df = df[(df['Date']>= start_date) & (df['Date']<= end_date)]
        fig= px.line(_7days_df, x = 'Date' , y = 'Close')
        st.plotly_chart(fig)
        
    elif button=='Last 1 Month':
        end_date=df['Date'].max()
        start_date= end_date - timedelta(30)
        _30days_df = df[(df['Date'] >= start_date) & (df['Date']<= end_date)]
        fig = px.line(_30days_df, x = 'Date' , y = 'Close')
        st.plotly_chart(fig)
        
    elif button=='Last 6 Months':
        end_date=df['Date'].max()
        start_date= end_date - timedelta(180)
        _180days_df=df[(df['Date']>=start_date) & (df['Date']<=end_date)]
        fig= px.line(_180days_df, x = 'Date' , y = 'Close' )
        st.plotly_chart(fig)
        
    elif button=='Last year':
        end_date=df['Date'].max()
        start_date= end_date - timedelta(365)
        _365days_df=df[(df['Date']>=start_date) & (df['Date']<=end_date)]
        fig= px.line(_365days_df, x = 'Date' , y = 'Close' )
        st.plotly_chart(fig)
        
    else:
        fig= px.line(df,x='Date', y='Close')
        st.plotly_chart(fig)
        
    st.subheader('Forcasting for next one year (2021 to 2022) ')
    dfp=pd.read_csv('Reliance_dataset (2022).csv')
    df1=pd.read_csv('Reliance_dataset (2022).csv')
    
    dfp.drop('Date',axis=1,inplace=True)
    dfp.drop('Volume',axis=1,inplace=True)
    
    close_stock = dfp.copy()
    scaler=MinMaxScaler(feature_range=(0,1))
    closedf=scaler.fit_transform(np.array(dfp).reshape(-1,1))
   
    # reshape into X=t,t+1,t+2,t+3 and Y=t+4
    time_step = 13
    X_test, y_test = create_dataset(closedf, time_step)
    
    # Lets Do the prediction
    test_predict=loaded_LSTM.predict(X_test)
    
   
    closedf = pd.DataFrame(closedf)  # Convert closedf to a DataFrame

    # Transform back to original form
    test_predict = scaler.inverse_transform(test_predict)
  

    closedf['Prediction'] = np.nan  # Assign np.nan to the 'Prediction' column

    # Assign the predicted values to the 'Prediction' column
    closedf.loc[time_step+1:len(closedf)-1, 'Prediction'] = test_predict.flatten()
    
    closedf['Date'] = df1['Date']
    
    fig= px.line(closedf, x = 'Date' , y = 'Prediction' )
    st.plotly_chart(fig)


else:
    st.title('Reliance Industries Ltd.')
    st.subheader('Company Profile')
    st.write('SECTOR : ' + 'Reliance Industries Limited operates in various sectors.\nThe company is primarily involved in the energy, petrochemicals,\nrefining,oil and gas exploration, and telecommunications sectors.')
    st.write('INDUSTRY : ' + 'The industry type of Reliance Industries Limited is conglomerate.')
    st.write('PHONE : '+ '+91-22-3555-5000')
    st.write('ADDRESS : '+ 'Reliance Industries Limited\nMaker Chambers - IV \nNariman Pointn\nMumbai 400 021, India')
    st.write('WEBSITE : '+ 'https://www.ril.com/')
    st.write('Business Summary')
    with open('summry.txt', 'r') as file:
       summry = file.read()
    st.text_input(summry)

        
    start = dt.datetime.today()-dt.timedelta(23*365)
    st.write(start)
    end = dt.datetime.today()
    st.write(end)
    df = yf.download('RELIANCE.NS',start,end)
    df = df.reset_index()
    fig = go.Figure(
            data=go.Scatter(x=df['Date'], y=df['Close'])
        )
    fig.update_layout(
        title={
            'text': "Stock Prices Over Past Years",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
    st.plotly_chart(fig, use_container_width=True)
    
st.sidebar.subheader('For other States of Stock on perticuler dates')
#Date input from user
min_date = datetime(2000, 1, 1).date() 
max_date = datetime(2022, 12, 31).date()
default_date= datetime(2021, 12, 18)
# Display a date input widget in the sidebar with limited dates
selected_date = st.sidebar.date_input('Select a date', min_value=min_date, max_value=max_date, value=default_date )
#selected_date = datetime(selected_date)
selected_date = datetime.combine(selected_date, datetime.min.time())

# Display the selected date in the main content area
# Filter the DataFrame based on the selected date
filtered_data = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']][data['Date'] == selected_date]

# Check if the filtered DataFrame is not empty before accessing its values
if not filtered_data.empty:
    for column in filtered_data.columns:
        value = filtered_data[column].values[0]
        st.sidebar.text(f"{column}: {value}")
else:
    st.sidebar.text("No data available for the selected date.")
   
