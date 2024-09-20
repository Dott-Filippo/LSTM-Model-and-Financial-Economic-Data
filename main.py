import random
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import pandas_datareader.data as pdr
import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.layers import Dropout
import plotly.graph_objs as go  

# Imposta il seed per NumPy
np.random.seed(42)

# Imposta il seed per Python
random.seed(42)

# Imposta il seed per TensorFlow
tf.random.set_seed(42)

def analizza_fasi(ticker):
    azienda = yf.Ticker(ticker)
    flussi_cassa = azienda.cashflow.T

    flussi_cassa.columns = flussi_cassa.columns.str.strip()

    try:
        flussi_cassa = flussi_cassa[['Operating Cash Flow', 'Investing Cash Flow', 'Financing Cash Flow']]
    except KeyError as e:
        print(f"Errore: {e}. Verifica i nomi delle colonne disponibili.")
        return None

    flussi_cassa.sort_index(ascending=False, inplace=True)
    last_3_years = flussi_cassa.head(3)

    fasi = pd.DataFrame(columns=['ds', 'fase'])

    for index, row in last_3_years.iterrows():
        operativo = row['Operating Cash Flow']
        investimento = row['Investing Cash Flow']
        finanziamento = row['Financing Cash Flow']

        if operativo > 0 and investimento < 0 and finanziamento > 0:
            fase = 2  # Crescita
        elif operativo < 0 and investimento < 0 and finanziamento > 0:
            fase = 1  # Introduzione
        elif operativo > 0 and investimento < 0 and finanziamento < 0:
            fase = 3  # Maturità
        elif (operativo < 0 and investimento < 0 and finanziamento < 0) or \
             (operativo > 0 and investimento > 0 and finanziamento > 0) or \
             (operativo > 0 and investimento > 0 and finanziamento < 0):
            fase = 4  # Shake-out
        elif operativo < 0 and investimento > 0 and finanziamento < 0:
            fase = 5  # Declino
        else:
            fase = 0  # N/A

        year = index.year
        date_range = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq='D')
        df_temp = pd.DataFrame({'ds': date_range, 'fase': fase})
        fasi = pd.concat([fasi, df_temp], ignore_index=True)

    fasi['ds'] = pd.to_datetime(fasi['ds'])
    fasi.sort_values(by='ds', inplace=True)

    today = pd.to_datetime("today").normalize()
    all_dates = pd.date_range(start=fasi['ds'].min(), end=today, freq='D')

    fasi = fasi.set_index('ds').reindex(all_dates, method='ffill').reset_index()
    fasi.columns = ['ds', 'fase']

    # Filtra le fasi per includere solo dal 2021
    fasi = fasi[fasi['ds'] >= '2021-01-01']

    return fasi

def get_financial_data(ticker):
    stock_data = yf.download(ticker, start='2021-01-01', end=datetime.now().strftime('%Y-%m-%d'))
    return stock_data[['Close']]

def get_economic_data():
    start_date = '2021-01-01'  # Modifica la data di inizio
    end_date = datetime.now().strftime('%Y-%m-%d')

    unemployment_rate = pdr.get_data_fred('UNRATE', start_date, end_date)
    inflation_rate = pdr.get_data_fred('CPIAUCSL', start_date, end_date)
    interest_rate = pdr.get_data_fred('FEDFUNDS', start_date, end_date)

    economic_data = pd.concat([unemployment_rate, inflation_rate, interest_rate], axis=1)
    economic_data.columns = ['Unemployment', 'Inflation', 'Interest']

    # Filtra i dati economici per includere solo dal 2021
    economic_data = economic_data[economic_data.index >= '2021-01-01']

    return economic_data

def preprocess_financial_data(price_data):
    scaler_price = MinMaxScaler(feature_range=(0, 1))
    scaled_price = scaler_price.fit_transform(price_data)
    return scaled_price, scaler_price

def preprocess_economic_data(economic_data):
    economic_data = economic_data.interpolate(method='linear', limit_direction='forward', axis=0)
    economic_data = economic_data.fillna(method='ffill')
    scaler_economic = MinMaxScaler(feature_range=(0, 1))
    scaled_economic = scaler_economic.fit_transform(economic_data)
    return scaled_economic, scaler_economic

def create_lstm_dataset(price_data, economic_data, phase_data, time_step=60):
    X, y = [], []
    for i in range(len(price_data) - time_step - 1):
        price_segment = price_data[i:(i + time_step)]
        economic_segment = economic_data[i:(i + time_step)]
        phase_segment = phase_data[i:(i + time_step)]
        X.append(np.hstack((price_segment, economic_segment, phase_segment)))
        y.append(price_data[i + time_step, 0])
    return np.array(X), np.array(y)



def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))  # Dropout per prevenire overfitting
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))  # Dropout per prevenire overfitting
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def predict_future(model, scaled_price, scaled_economic, scaled_phase, time_step, future_days, num_predictions=2):
    all_predictions = []

    for _ in range(num_predictions):
        last_data = np.hstack((scaled_price[-time_step:], scaled_economic[-time_step:], scaled_phase[-time_step:]))
        future_predictions = []

        for _ in range(future_days):
            last_data_reshaped = last_data.reshape((1, time_step, last_data.shape[1]))
            prediction = model.predict(last_data_reshaped)
            future_predictions.append(prediction[0, 0])

            new_data = np.roll(last_data, shift=-1, axis=0)
            new_data[-1, 0] = prediction[0, 0]

            last_data = new_data

        all_predictions.append(future_predictions)

    # Media delle previsioni
    averaged_predictions = np.mean(all_predictions, axis=0)

    return averaged_predictions




def get_full_financial_data(ticker):
    # Recupera i dati storici a partire dal 1990
    stock_data_full = yf.download(ticker, start='1990-01-01', end=datetime.now().strftime('%Y-%m-%d'))
    return stock_data_full[['Close']]




def main():
    st.title("Firm Life Cycle and LSTM Model: a predictive analysis")

    # Aggiungi l'avviso per la versione beta e disclaimer
    st.warning("⚠️ **Versione Beta**: Questo strumento è in fase sperimentale. Le previsioni potrebbero essere imprecise e non devono essere considerate come consigli finanziari.")
    
    visualdata = False
    # Input per il ticker
    ticker = st.text_input("Inserisci il ticker dell'azienda:", value="AAPL")
    if st.checkbox("Mostrai dati"): 
        visualdata = True
    if st.button("Analizza"):
        with st.spinner("Elaborazione in corso... Attendere, potrebbero volerci alcuni minuti"):
            # Esegui le funzioni esistenti
            phase_data = analizza_fasi(ticker)
            price_data = get_financial_data(ticker)
            economic_data = get_economic_data()

            if price_data.empty:
                st.error("Impossibile recuperare i dati per il ticker fornito.")
                return

            # Mostra i dati
            if visualdata == True:
               st.subheader("Dati storici dei prezzi:")
               st.write(price_data.head())
               st.write(price_data.tail())

               st.subheader("Dati economici:")
               st.write(economic_data.head())
               st.write(economic_data.tail())

               st.subheader("Dati delle fasi:")
               st.write(phase_data.head())
               st.write(phase_data.tail())

            # Combina i dati
            combined_data = price_data.join(economic_data, how='outer')
            combined_data.interpolate(method='linear', inplace=True)
            phase_data.set_index('ds', inplace=True)
            combined_data = combined_data.join(phase_data, how='outer')
            combined_data.dropna(inplace=True)

            # Controlla che i dati non siano vuoti
            if combined_data.empty:
                st.error("Errore: Dopo la creazione del dataset, i dati risultano vuoti.")
                return

            # Preprocessing dei dati
            phase_array = combined_data['fase'].values.reshape(-1, 1)
            scaler_phase = MinMaxScaler(feature_range=(0, 1))
            scaled_phase = scaler_phase.fit_transform(phase_array)
            scaled_price, scaler_price = preprocess_financial_data(combined_data[['Close']])
            scaled_economic, scaler_economic = preprocess_economic_data(combined_data[['Unemployment', 'Inflation', 'Interest']])

            # Crea il dataset per l'allenamento LSTM
            time_step = 60
            X, y = create_lstm_dataset(scaled_price, scaled_economic, scaled_phase, time_step)

            # Controlla che i dataset non siano vuoti dopo la creazione del dataset
            if X.size == 0 or y.size == 0:
                st.error("Errore: Dopo la creazione del dataset, i dati risultano vuoti.")
                return

            # Addestra il modello LSTM
            X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))
            model = build_lstm_model((X.shape[1], X.shape[2]))
            model.fit(X, y, epochs=20, batch_size=32, verbose=1)

            # Previsione futura
            future_end_date = datetime(2030, 1, 1)
            future_days = (future_end_date - combined_data.index[-1]).days
            averaged_predictions = predict_future(model, scaled_price, scaled_economic, scaled_phase, time_step, future_days)

            future_dates = [combined_data.index[-1] + timedelta(days=i) for i in range(1, future_days + 1)]
            averaged_predictions = scaler_price.inverse_transform(np.array(averaged_predictions).reshape(-1, 1))

            # Combina i prezzi storici e i prezzi predetti in un'unica serie
            all_prices = np.concatenate((combined_data['Close'].values, averaged_predictions.flatten()))
            all_dates = np.concatenate((combined_data.index.values, future_dates))

            # Grafico interattivo con Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=combined_data.index, y=combined_data['Close'], mode='lines', name='Dati Reali', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x= all_dates, y=all_prices, mode='lines', name='Previsioni LSTM', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode='lines', name='Previsioni LSTM', line=dict(color='red')))
            fig.update_layout(title=f'Previsione del Prezzo delle Azioni per {ticker} con LSTM',
                              xaxis_title='Data',
                              yaxis_title='Prezzo di Chiusura',
                              hovermode='x unified')
  
            # Visualizza il grafico in Streamlit
            st.plotly_chart(fig)

            # Aggiungi la fase attuale con parole descrittive
            current_phase_num_print = phase_data['fase'].iloc[-1] if not phase_data.empty else None
            phase_description = {
                0: "N/A",
                1: "Introduzione",
                2: "Crescita",
                3: "Maturità",
                4: "Shake-out",
                5: "Declino"
            }
            current_phase_desc = phase_description.get(current_phase_num_print, "Sconosciuta")

            # Aggiungi la descrizione dell'andamento
            current_phase_num_print_esplit = phase_data['fase'].iloc[-1] if not phase_data.empty else None
            phase_description_esplit = {
                0: "N/A",
                1: "In questa fase, l'azienda è appena entrata nel mercato. Il prodotto o servizio è nuovo, l'azienda sta cercando di costruire una base di clienti e aumentare la consapevolezza del marchio.",
                2: "In questa fase, l'azienda sta crescendo rapidamente. La domanda per i prodotti o servizi è in aumento, e la base clienti si sta espandendo rapidamente.",
                3: "In questa fase, l'azienda ha raggiunto una posizione consolidata nel mercato. La crescita dei ricavi rallenta, ma i margini di profitto sono solidi.",
                4: "In questa fase, l'azienda potrebbe affrontare difficoltà nel mercato a causa di una riduzione della domanda o di cambiamenti nel settore. La concorrenza si intensifica e alcune aziende potrebbero uscire dal mercato.",
                5: "In questa fase, l'azienda affronta un forte calo della domanda dei suoi prodotti o servizi. Il settore potrebbe essere diventato obsoleto o la concorrenza ha preso il sopravvento."
            }
            current_phase_desc_esplit = phase_description_esplit.get(current_phase_num_print_esplit, "Sconosciuta")

            if current_phase_num_print is not None:
                st.subheader("Fase attuale dell'azienda")
                st.write(f"L'azienda si trova nella fase **{current_phase_num_print}** ({current_phase_desc}) del suo ciclo di vita. {current_phase_desc_esplit}")
            else:
                st.write("Non è stato possibile determinare la fase attuale dell'azienda.")

if __name__ == "__main__":
    main()
