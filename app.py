import streamlit as st
import pandas as pd
import altair as alt

from training import df, le_model, le_region, le_color, le_fuel, le_transmission

import pickle

df_1 = pd.read_csv("BMW_Car_Sales_Classification.csv", dtype = {"Year": str})

st.write("BMW Sales Price Prediction")

model_name = st.text_input("Enter the name of the model:")

regions = df_1.Region.unique()
region = st.selectbox("Choose the region of sale:", options = regions)

color = st.text_input("Enter the color of the car:")

transmission_types = df_1.Transmission.unique()
transmission = st.selectbox("Choose type of transmission:", options =  transmission_types)

fuel_types = df_1.Fuel_Type.unique()
fuel = st.selectbox("Choose type of fuel:", options = fuel_types)

engine_size = st.number_input("Enter the size of the engine (in litres):")

def find(colname, dataframe, value):
    df = dataframe[dataframe[colname] == value]
    return df

def predict_price(row):
    colnames = ['Model', 'Region', 'Fuel_Type', 'Engine_Size_L', 'Transmission', 'Color']
    temp = df
    charts = []

    for col in colnames:
        temp1 = find(col, temp, row[col])
        if not temp1.empty:
            temp_df = df[df[col] == row[col]][['Year', 'Price_USD']]
            sum_df = temp_df.groupby('Year').mean().reset_index()
            sum_df['Year'] = sum_df['Year'].astype(str)
            df_melted = pd.melt(sum_df, id_vars = ['Year'], var_name = 'parameter', value_name = 'value')
            c = alt.Chart(df_melted, title = f"Price trend for {row[col]} {col} over the years").mark_line().encode(x = 'Year', y = 'value', color = 'parameter')
            charts.append(c)
            temp = temp1

    pop_score = temp.sort_values(by = 'Year', ascending = False).head(5)['Popularity_Score'].mean()
    row['Popularity_Score'] = pop_score

    if row['Model'] in df_1.Model.values:
        row['Model'] = le_model.transform([row['Model']])

    else:
        row['Model'] = -1

    row['Region'] = le_region.transform([row['Region']])
    if row['Color'] in df_1.Color.values:
        row['Color'] = le_color.transform([row['Color']])

    else:
        row['Color'] = -1

    row['Fuel_Type'] = le_fuel.transform([row['Fuel_Type']])
    row['Transmission'] = le_transmission.transform([row['Transmission']])

    model = pickle.load(open('model_weights.pickle', 'rb'))
    
    price = model.predict(pd.DataFrame(row))

    return price * 100000, charts

def process():
    row = {'Model': model_name, 'Region': region, 'Color': color, 'Transmission': transmission, 'Fuel_Type': fuel, 'Engine_Size_L': float(engine_size)}

    price, charts = predict_price(row)

    st.session_state['predicted_price'] = round(price[0], 2)

    st.session_state['plots'] = charts

st.button("Submit info", on_click = process)

result_placeholder = st.empty()
plot_placeholder = st.empty()

if 'predicted_price' in st.session_state:
    result_placeholder.write(f"Predicted price of the car: $ {st.session_state['predicted_price']}")

if 'plots' in st.session_state:
    for chart in st.session_state['plots']:
        st.altair_chart(chart, use_container_width = True, theme = "streamlit")