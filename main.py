import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score    
import plotly.express as px

# Synthetic generation of data examples for training the model
def generate_house_data(n_samples=100):
    np.random.seed(42)
    size = np.random.normal(1500, 500, n_samples)
    price = size * 100 + np.random.normal(0, 10000, n_samples)
    return pd.DataFrame({'Surface(m2)': size, 'Prix' : price})

# Function for instantiating and training linear regression model
def train_model():
    df = generate_house_data()

    # Train-test data splitting
    X = df[['Surface(m2)']]
    y = df['Prix']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model

# Streamlit user Interface for deployed model
def main():
    st.title('&#127968;' 'Application de prédiction des prix immobiliers')
    st.write('Saississez la taille de la maison en pieds carrés pour prédire le prix estimé.')

    # Train model
    model = train_model()

    # User input
    size = st.number_input('Surface de la maison (mètres carré)', min_value=500, max_value=5000, value=1500)

    if st.button('Prédire le prix'):
        # Perform prediction
        # prediction = model.predict([[size]])

        input_df = pd.DataFrame({'Surface(m2)': [size]})
        prediction = model.predict(input_df)

        # Show result
        st.success(f'Prix estimé : ${prediction[0] : ,.2f}')

        # Visualization
        df = generate_house_data()
        fig = px.scatter(df, x = 'Surface(m2)', y = 'Prix', title = 'Surface vs Prix')
        fig.add_scatter(x = [size], y = [prediction[0]], mode = 'markers', marker = dict(size=15, color =  'red'), name = 'Prediction')
        st.plotly_chart(fig)

if __name__ == '__main__':
    main()