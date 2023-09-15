import streamlit as st
import requests
import pandas as pd
import numpy as np
import datarobotx as drx
from PIL import Image
from io import StringIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime as dt
import openai
openai.api_key = "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"
# Set the maximum number of rows and columns to be displayed
pd.set_option('display.max_rows', None)     # Display all rows
pd.set_option('display.max_columns', None)  # Display all columns
import logging
#Configure the page title, favicon, layout, etc
st.set_page_config(page_title="Financial Plannning and Analysis",
                   layout="wide")

API_URL = 'https://cfds-ccm-prod.orm.datarobot.com/predApi/v1.0/deployments/{deployment_id}/predictions?passthroughColumnsSet=all'    # noqa
API_KEY = st.secrets['DATAROBOT_API_TOKEN']

DATAROBOT_KEY = st.secrets['DATAROBOT_KEY']
forecast_point=None


@st.cache_data(show_spinner=False)
def scoreNowcast(data, deployment_id):
    #read training data for the purpose of displaying historical training data
    train = pd.read_csv(r"TrainNOX.csv")
    train["Segment"] = "History"
    score = data
    score["Segment"] = "History"
    predictions_start_date = score["Datetime"].min()
    predictions_end_date = score["Datetime"].max()

    headers = {
        'Content-Type': 'application/json; charset=UTF-8',
        'Authorization': 'Bearer {}'.format(API_KEY),
        'Accept': 'text/csv',
        'DataRobot-Key': DATAROBOT_KEY,
    }

    url = API_URL.format(deployment_id=deployment_id)

    params = {
        #'forecastPoint': forecast_point,
        'predictionsStartDate': predictions_start_date,
        'predictionsEndDate': predictions_end_date,
        # If explanations are required, uncomment the line below
        #  'maxExplanations': 3,
        #  'thresholdHigh': 0.5,
        #  'thresholdLow': 0.15,
        # Uncomment this for Prediction Warnings, if enabled for your deployment.
        # 'predictionWarningEnabled': 'true',
    }

    # Make API request for predictions
    predictions_response = requests.post(url, data=score.to_json(orient='records'), headers=headers, params=params)
    results = StringIO(predictions_response.content.decode("utf-8"))
    dfResults = pd.read_csv(results, sep=",")
    # dfResults["Segment"] = "Forecast"
    # dfResults["Datetime"] = pd.to_datetime(dfResults["Datetime"])
    # dfResults = pd.concat([train, score, dfResults], axis=0)
    # dfResults.reset_index(drop=True, inplace=True)
    #dfResults.loc[dfResults["SALESQTY"].isna(), "SALESQTY"] = dfResults["SALESQTY (actual)_PREDICTION"]



    # dfResults["Series Type"] = "Full History"
    # dfResults.loc[dfResults["SERIES"].isin(WARM_STARTS), "Series Type"] = "Warm Start"
    # dfResults.loc[dfResults["SERIES"].isin(COLD_STARTS), "Series Type"] = "Cold Start"

    return dfResults

@st.cache_data(show_spinner=False)
def scoreOTV(data, deployment_id):
    #read training data for the purpose of displaying historical training data
    train = pd.read_csv(r"TrainNOX.csv")
    train["Segment"] = "History"
    score = data
    score["Segment"] = "History"
    predictions_start_date = score["Datetime"].min()
    predictions_end_date = score["Datetime"].max()

    headers = {
        'Content-Type': 'application/json; charset=UTF-8',
        'Authorization': 'Bearer {}'.format(API_KEY),
        'Accept': 'text/csv',
        'DataRobot-Key': DATAROBOT_KEY,
    }

    url = API_URL.format(deployment_id=deployment_id)

    params = {
        # If explanations are required, uncomment the line below
        #  'maxExplanations': 3,
        #  'thresholdHigh': 0.5,
        #  'thresholdLow': 0.15,
        # Uncomment this for Prediction Warnings, if enabled for your deployment.
        # 'predictionWarningEnabled': 'true',
    }

    # Make API request for predictions
    predictions_response = requests.post(url, data=score.to_json(orient='records'), headers=headers, params=params)
    results = StringIO(predictions_response.content.decode("utf-8"))
    dfResults = pd.read_csv(results, sep=",")
    # dfResults["Segment"] = "Forecast"
    # dfResults["Datetime"] = pd.to_datetime(dfResults["Datetime"])
    # dfResults = pd.concat([train, score, dfResults], axis=0)
    # dfResults.reset_index(drop=True, inplace=True)
    #dfResults.loc[dfResults["SALESQTY"].isna(), "SALESQTY"] = dfResults["SALESQTY (actual)_PREDICTION"]



    # dfResults["Series Type"] = "Full History"
    # dfResults.loc[dfResults["SERIES"].isin(WARM_STARTS), "Series Type"] = "Warm Start"
    # dfResults.loc[dfResults["SERIES"].isin(COLD_STARTS), "Series Type"] = "Cold Start"

    return dfResults

@st.cache_data(show_spinner=False)
def scoreCV(data, deployment_id):
    #read training data for the purpose of displaying historical training data
    train = pd.read_csv(r"TrainNOX.csv")
    train["Segment"] = "History"
    score = data
    score["Segment"] = "History"
    predictions_start_date = score["Datetime"].min()
    predictions_end_date = score["Datetime"].max()

    headers = {
        'Content-Type': 'application/json; charset=UTF-8',
        'Authorization': 'Bearer {}'.format(API_KEY),
        'Accept': 'text/csv',
        'DataRobot-Key': DATAROBOT_KEY,
    }

    url = API_URL.format(deployment_id=deployment_id)

    params = {
        # If explanations are required, uncomment the line below
        #  'maxExplanations': 3,
        #  'thresholdHigh': 0.5,
        #  'thresholdLow': 0.15,
        # Uncomment this for Prediction Warnings, if enabled for your deployment.
        # 'predictionWarningEnabled': 'true',
    }

    # Make API request for predictions
    predictions_response = requests.post(url, data=score.to_json(orient='records'), headers=headers, params=params)
    results = StringIO(predictions_response.content.decode("utf-8"))
    dfResults = pd.read_csv(results, sep=",")
    # dfResults["Segment"] = "Forecast"
    # dfResults["Datetime"] = pd.to_datetime(dfResults["Datetime"])
    # dfResults = pd.concat([train, score, dfResults], axis=0)
    # dfResults.reset_index(drop=True, inplace=True)
    #dfResults.loc[dfResults["SALESQTY"].isna(), "SALESQTY"] = dfResults["SALESQTY (actual)_PREDICTION"]



    # dfResults["Series Type"] = "Full History"
    # dfResults.loc[dfResults["SERIES"].isin(WARM_STARTS), "Series Type"] = "Warm Start"
    # dfResults.loc[dfResults["SERIES"].isin(COLD_STARTS), "Series Type"] = "Cold Start"

    return dfResults

@st.cache_data(show_spinner=False)
def scoreForecast(df):
    #df = pd.read_csv(r"Retail Sales Data 4.csv")
    deployment = drx.Deployment(deployment_id='64ff43e6e3901a43859cd9f9')  # Get deployment
    # Get predictions with explanations
    predictions_with_explanations = deployment.predict(df, max_explanations=6)  # Predictions with explanations

    # Reset the index of predictions_with_explanations and drop the old index
    predictions_with_explanations.reset_index(inplace=True, drop=True)

    # Add a new 'row_id' column based on the DataFrame index
    predictions_with_explanations['row_id'] = predictions_with_explanations.index

    # Melt the DataFrame using datarobotx function
    pdf_melted = drx.melt_explanations(predictions_with_explanations)

    # Merge two DataFrames based on the 'row_id' column
    pdf_joined = pd.merge(
        predictions_with_explanations[['row_id', 'seriesId', 'forecastPoint']],
        pdf_melted, how='outer', on='row_id'
    )

    # Group the merged DataFrame and aggregate the 'strength' column by sum
    pdf_joined_grouped = pdf_joined.groupby(['seriesId', 'forecastPoint', 'feature_name'])['strength'].agg(
        ['sum']).reset_index()

    # Calculate 'Cumulative Strength Direction' based on the 'sum' column
    pdf_joined_grouped['Cumulative Strength Direction'] = np.where(pdf_joined_grouped['sum'] >= 0, 'positive',
                                                                   'negative')

    # Take the absolute value of the 'sum' column
    pdf_joined_grouped['sum'] = pdf_joined_grouped['sum'].abs()

    # Sort the DataFrame by multiple columns
    pdf_joined_grouped = pdf_joined_grouped.sort_values(by=['seriesId', 'forecastPoint', 'sum'],
                                                        ascending=[True, True, False])

    # Select the top 5 rows for each group and reset the index
    pdf_joined_grouped_top = pdf_joined_grouped.groupby(['seriesId', 'forecastPoint']).head(6).reset_index(drop=True)

    # Define a mapping for column names
    column_mapping = {
        'seriesId': "Series",
        'forecastPoint': 'Forecast Starting Point',
        'feature_name': 'Feature Name',
        'sum': 'Cumulative Strength'
    }

    # Rename columns using the defined mapping
    pdf_joined_grouped_top = pdf_joined_grouped_top.rename(columns=column_mapping)
    return pdf_joined_grouped_top, predictions_with_explanations.copy()

@st.cache_data(show_spinner=False)
def createChart(history, forecast, title):

    # Create the Chart
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    fig.add_trace(
        go.Scatter(
            x=history["Snap_Date"],
            y=history["Week_Sales_SPT"],
            mode="lines",
            name="Sales History",
            line_shape="spline",
            line=dict(color="#ff9e00", width=2)
        ))
    fig.add_trace(
        go.Scatter(
            x=forecast["timestamp"],
            y=forecast["PREDICTION_75_PERCENTILE_LOW"],
            mode="lines",
            name="Low forecast",
            line_shape="spline",
            line=dict(color="#335599", width=0.5,dash="dot")
        ))
    fig.add_trace(
        go.Scatter(
            x=forecast["timestamp"],
            y=forecast["PREDICTION_75_PERCENTILE_HIGH"],
            mode="lines",
            name="High forecast",
            line_shape="spline",
            line=dict(color="#335599", width=0.5,dash="dot")
        ))
    fig.add_trace(
        go.Scatter(
            x=forecast["timestamp"],
            y=forecast["prediction"],
            mode="lines",
            name="1-Week Total Sales Forecast",
            line_shape="spline",
            line=dict(color="#162955", width=2)
        ))

    fig.add_vline(x=history["Snap_Date"].max(), line_width=2, line_dash="dash", line_color="gray")


    fig.update_xaxes(
        color="#404040",
        title_font_family="Gravitas One",
        title_text="Date",
        linecolor="#adadad",
    )
    fig.update_yaxes(
        color="#404040",
        title_font_size=16,
        title_text="Dollars ($)",
        linecolor="#adadad",
        gridcolor="#f2f2f2",
    )

    fig.update_layout(
        height=600,
        title=title,
        title_font_size=20,
        hovermode="x unified",
        plot_bgcolor="#ffffff",
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.5),
        margin=dict(l=50, r=50, b=20, t=50, pad=4),
        xaxis=dict(rangeslider=dict(visible=True), type="date"),
        uniformtext_mode='hide'
    )

    fig.update_layout(
        xaxis=dict(fixedrange=False), yaxis=dict(fixedrange=False)
    )
    fig.update_traces(connectgaps=False)
    config = {"displayModeBar": False, "responsive": True}



    st.plotly_chart(fig, config=config, use_container_width=True)

def interpretChart(explanations, forecast):
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        #model="gpt-3.5-turbo",
        temperature=0.9,
        messages=[
            {"role": "system",
             "content": """
                        You are a data analyst. Explain to non-technical executives business leaders what this data suggests.
                        Executive leadership will provide a sales forecast and you will interpret the outlook and highlight key insights.                         
                        In addition to the forecast, a time series analysis has been performed to identify the key drivers of the sales forecast.
                        These key drivers are summarised in the explanations table. Only discuss the most important drivers of the sales forecast from the explanations table.
                        Start by noting the overall trend. Mention the most important observations of the forecast and explanations.
                        Provide a few bullets calling out key insights, trends, any seasonality, possible anomalies, and anything noteworthy.
                        Complex feature names should be should be transformed into plain english. 
                        For example: Instead of saying "Week_Sales_SPT (12 week median)" you could say "12 week median sales".
                        Another example would be: Instead of saying "Snap_Date (Month) (actual)" you should just say "Date".
                        Don't use underscores. For example: If discussing the 'Tool_Brand' feature, just say Tool Brand.                         
                        Don't reference "tables". For example: Instead of saying "the forecast table", just say "the forecast".
                        When referring to lags, say "weeks ago" instead. For Example: Instead of saying "Tool Brand sales at a 1-week lag",
                        say, "Tool Brand sales from 1 week ago".  
                        Overall, your explanation should be no more than 8 sentences.
                        Even if you're unsure, speak with confidence and certainty.
                                                
                        
                        """},
            {"role": "user", "content": "Forecast:"
             +str(forecast[["timestamp","prediction"]])
             +"Explanations:"
             +str(explanations)}
        ]
    )
    return completion.choices[0].message.content


def interpretChartHeadline(forecast):
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        #model="gpt-3.5-turbo",
        temperature=0.2,
        messages=[
            {"role": "system",
             "content": """
                        You are a data analyst. Explain to non-technical executives business leaders what this data suggests.
                        Executive leadership will provide a sales forecast and you will interpret it and summarize the outlook, highlighting key insights.                         
                        Your response should be only 1 sentence long, not very wordy. It should be like a news headline. Do not put quotation marks around it.
                        Your response, while insightful, should speak to the general direction of the forecast.
                        Even if you're unsure, speak with confidence and certainty.
                        """},
            {"role": "user", "content": "Forecast:"
                                        + str(forecast[["timestamp", "prediction"]])
                                        }
        ]
    )
    return completion.choices[0].message.content

def processForecast(df, level):
    explanations, forecast = scoreForecast(df)
    # Replace the hyphen with an asterisk in "Access-Indus (THA)"
    forecast['seriesId'] = forecast['seriesId'].str.strip().str.replace('s-I', 's*I')
    explanations['Series'] = explanations['Series'].str.strip().str.replace('s-I', 's*I')

    # Create new columns for the levels forecast
    seriesIdComponentsForecast = forecast['seriesId'].str.strip().str.split('-', expand=True)
    seriesIdComponentsForecast = seriesIdComponentsForecast.iloc[:, :-1]
    seriesIdComponentsForecast.columns = ['Store_Name', 'Store_Division', 'Tool_Brand', 'Major_Business_Unit', 'Tool_Category']

    # Create new columns for the levels explanations
    seriesIdComponentsExplanations = explanations['Series'].str.strip().str.split('-', expand=True)
    seriesIdComponentsExplanations = seriesIdComponentsExplanations.iloc[:, :-1]
    seriesIdComponentsExplanations.columns = ['Store_Name', 'Store_Division', 'Tool_Brand', 'Major_Business_Unit','Tool_Category']

    # Replace the asterisk back to a hyphen
    seriesIdComponentsForecast['Store_Name'] = seriesIdComponentsForecast['Store_Name'].str.replace('*', '-')
    seriesIdComponentsExplanations['Store_Name'] = seriesIdComponentsExplanations['Store_Name'].str.replace('*', '-')

    # Merge with the original Dataframe
    forecast = pd.concat([forecast, seriesIdComponentsForecast], axis=1)
    forecast['timestamp'] = pd.to_datetime(forecast['timestamp'])
    explanations = pd.concat([explanations, seriesIdComponentsExplanations], axis=1)
    explanations['Forecast Starting Point'] = pd.to_datetime(explanations['Forecast Starting Point'])


    # Create forecast at the level
    levelForecast = forecast.groupby([level, pd.Grouper(key='timestamp', freq='W')]).sum().reset_index()
    levelHistory = df.groupby([level, pd.Grouper(key='Snap_Date', freq='W')]).sum().reset_index()
    levelExplanations = explanations.groupby([level, pd.Grouper(key='Forecast Starting Point', freq='W')]).sum().reset_index()
    return levelForecast, levelExplanations, levelHistory

def fpa():
    # Layout
    titleContainer = st.container()
    col1, col2, col3 = titleContainer.columns([1,1,1])
    headerContainer = st.container()
    headlineContainer = st.container()
    chartContainer = st.container()
    explanationContainer = st.container()

    # Header
    with titleContainer:
        col1.image("datarobotlogo.png", width=200)
        col2.header("Retail Tool Sales")


    # Load historical data for the select boxes and for scoring
    df = pd.read_csv("Retail Sales Data 4.csv", parse_dates=["Snap_Date"])
    df.sort_values(by=["Snap_Date"], ascending=True, inplace=True)

    # Setup dropdown menues in the sidebar
    with st.sidebar:
        retailers = df["Store_Name"].unique().tolist()
        retailers.insert(0, None)
        divisions = df["Store_Division"].unique().tolist()
        divisions.insert(0, None)
        toolBrands = df["Tool_Brand"].unique().tolist()
        toolBrands.insert(0, None)

        with st.form(key="sidebar_form"):
            retailer = st.selectbox("Retailer", options=retailers)
            division = st.selectbox("Division", options=divisions)
            toolBrand = st.selectbox("Tool Brand", options=toolBrands)
            sidebarSubmit = st.form_submit_button(label="Run Forecast")
            if sidebarSubmit:
                # Handle Store Level
                if retailer is not None and division is None and toolBrand is None:
                    # Clear charts
                    # headlineContainer = headlineContainer.empty()
                    # chartContainer = chartContainer.empty()
                    # explanationContainer = explanationContainer.empty()

                    # Execute the forecast
                    with st.spinner("Processing forecast..."):
                        retailerForecast, retailerExplanations, retailerHistory = processForecast(df=df, level="Store_Name")

                    with chartContainer:
                        chart = createChart(retailerHistory.loc[retailerHistory["Store_Name"] == retailer].tail(104), retailerForecast.loc[retailerForecast["Store_Name"] == retailer],"Sales for " + str(retailer))

                    with headlineContainer:
                        with st.spinner("Generating Headline..."):
                            headline = st.subheader(interpretChartHeadline(retailerForecast.loc[retailerForecast["Store_Name"] == retailer]))

                    with explanationContainer:
                        with st.spinner("Generating explanation..."):
                            st.write("**AI Generated Analysis:**")
                            detailedExplanation = explanationContainer.write(interpretChart(retailerExplanations.loc[retailerExplanations["Store_Name"] == retailer], retailerForecast.loc[retailerForecast["Store_Name"] == retailer]))

                # Handle Division Level
                if division is not None and toolBrand is None:
                    # Clear charts
                    # headlineContainer = headlineContainer.empty()
                    # chartContainer = chartContainer.empty()
                    # explanationContainer = explanationContainer.empty()

                    # Execute the forecast
                    with st.spinner("Processing forecast..."):
                        divisionForecast, divisionExplanations, divisionHistory = processForecast(df=df, level="Store_Division")

                    with chartContainer:
                        chart = createChart(divisionHistory.loc[divisionHistory["Store_Division"] == division].tail(104), divisionForecast.loc[divisionForecast["Store_Division"] == division],"Sales for " + str(division))

                    with headlineContainer:
                        with st.spinner("Generating Headline..."):
                            headline = st.subheader(interpretChartHeadline(divisionForecast.loc[divisionForecast["Store_Division"] == division]))

                    with explanationContainer:
                        with st.spinner("Generating explanation..."):
                            st.write("**AI Generated Analysis:**")
                            detailedExplanation = explanationContainer.write(interpretChart(divisionExplanations.loc[divisionExplanations["Store_Division"] == division], divisionForecast.loc[divisionForecast["Store_Division"] == division]))

                # Handle toolBrand Level
                if toolBrand is not None:
                    # Clear charts
                    # headlineContainer = headlineContainer.empty()
                    # chartContainer = chartContainer.empty()
                    # explanationContainer = explanationContainer.empty()

                    # Execute the forecast
                    with st.spinner("Processing forecast..."):
                        toolBrandForecast, toolBrandExplanations, toolBrandHistory = processForecast(df=df, level="Tool_Brand")

                    with chartContainer:
                        chart = createChart(
                            toolBrandHistory.loc[toolBrandHistory["Tool_Brand"] == toolBrand].tail(104),
                            toolBrandForecast.loc[toolBrandForecast["Tool_Brand"] == toolBrand],
                            "Sales for " + str(toolBrand))

                    with headlineContainer:
                        with st.spinner("Generating Headline..."):
                            headline = st.subheader(interpretChartHeadline(toolBrandForecast.loc[toolBrandForecast["Tool_Brand"] == toolBrand]))

                    with explanationContainer:
                        with st.spinner("Generating explanation..."):
                            st.write("**AI Generated Analysis:**")
                            detailedExplanation = explanationContainer.write(interpretChart(toolBrandExplanations.loc[toolBrandExplanations["Tool_Brand"] == toolBrand], toolBrandForecast.loc[toolBrandForecast["Tool_Brand"] == toolBrand]))



#Main app
def _main():
    hide_streamlit_style = """
    <style>
    # MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) # This let's you hide the Streamlit branding

    fpa()



if __name__ == "__main__":
    _main()


