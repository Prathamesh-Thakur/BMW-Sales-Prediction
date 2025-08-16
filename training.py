import numpy as np
import pandas as pd 

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error, mean_squared_error

import pickle

df = pd.read_csv("BMW_Car_Sales_Classification.csv")

df['Popularity_Score'] = np.nan

df['Price_USD'] = df['Price_USD'].astype(float, errors = 'ignore')
df['Sales_Volume'] = df['Sales_Volume'].astype(float, errors = 'ignore')
df['Mileage_KM'] = df['Mileage_KM'].astype(float, errors = 'ignore')

def find_pop_score(level2):
    transmission_props = level2.Transmission.value_counts(normalize = True)
    fuel_props = level2.Fuel_Type.value_counts(normalize = True)
    color_props = level2.Color.value_counts(normalize = True)
    engine_props = level2.Engine_Size_L.value_counts(normalize = True)
    scaler = MinMaxScaler()

    level2.loc[:, 'Price_USD'] = scaler.fit_transform(level2[['Price_USD']])
    level2.loc[:, 'Sales_Volume'] = scaler.fit_transform(level2[['Sales_Volume']])
    level2.loc[:, 'Mileage_KM'] = scaler.fit_transform(level2[['Mileage_KM']])

    for index, row in level2.iterrows():
        color = color_props[row['Color']]
        fuel_type = fuel_props[row['Fuel_Type']]
        transmission = transmission_props[row['Transmission']]
        engine_size = engine_props[row['Engine_Size_L']]
        mileage = row['Mileage_KM']

        pop_score = color * fuel_type * transmission * engine_size * row['Sales_Volume'] * row['Price_USD'] * mileage

        level2.loc[index, 'Popularity_Score'] = pop_score

    level2.loc[:, 'Popularity_Score'] = scaler.fit_transform(level2[['Popularity_Score']])

    return level2[['Popularity_Score']]

def populate_pop_scores():
    years = df['Year'].unique()

    for year in years:
        level1 = df[df['Year'] == year]

        regions = level1['Region'].unique()

        for region in regions:
            level2 = level1[level1['Region'] == region]

            updated_level2 = find_pop_score(level2)

            for index in updated_level2.index:
                df.loc[index, 'Popularity_Score'] = updated_level2.loc[index, 'Popularity_Score']


populate_pop_scores()

X = df[['Model', 'Region', 'Color', 'Transmission', 'Fuel_Type', 'Engine_Size_L', 'Popularity_Score']]
y = df['Price_USD'] / 100000

le_model = LabelEncoder()
le_region = LabelEncoder()
le_color = LabelEncoder()
le_fuel = LabelEncoder()
le_transmission = LabelEncoder()

le_model.fit(X['Model'])
le_region.fit(X['Region'])
le_color.fit(X['Color'])
le_fuel.fit(X['Fuel_Type'])
le_transmission.fit(X['Transmission'])

X.loc[:, 'Model'] = le_model.transform(X['Model'])
X.loc[:, 'Region'] = le_region.transform(X['Region'])
X.loc[:, 'Color'] = le_color.transform(X['Color'])
X.loc[:, 'Fuel_Type'] = le_fuel.transform(X['Fuel_Type'])
X.loc[:, 'Transmission'] = le_transmission.transform(X['Transmission'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

model = LinearRegression()
model.fit(X_train, y_train)

pickle.dump(model, open('model_weights.pickle', 'wb'))

preds = model.predict(X_test)

print("MAE of the model:", mean_squared_error(preds, y_test))
print("MSE of the model:", mean_absolute_error(preds, y_test))
