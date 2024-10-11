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
from PIL import Image
import plotly.graph_objs as go
from prophet import Prophet
from datetime import datetime
# Monkey patch per compatibilità con vecchi riferimenti a np.float_
np.float_ = np.float64

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


def predict_future(model, scaled_price, scaled_economic, scaled_phase, time_step, future_days):
    last_data = np.hstack((scaled_price[-time_step:], scaled_economic[-time_step:], scaled_phase[-time_step:]))
    future_predictions = []

    for _ in range(future_days):
        last_data_reshaped = last_data.reshape((1, time_step, last_data.shape[1]))
        prediction = model.predict(last_data_reshaped)
        future_predictions.append(prediction[0, 0])

        new_data = np.roll(last_data, shift=-1, axis=0)
        new_data[-1, 0] = prediction[0, 0]

        last_data = new_data

    return future_predictions


def get_full_financial_data(ticker):
    # Recupera i dati storici a partire dal 1990
    stock_data_full = yf.download(ticker, start='1990-01-01', end=datetime.now().strftime('%Y-%m-%d'))
    return stock_data_full[['Close']]

def get_financial_data_prophet(ticker):
    """Scarica i dati storici per un ticker specificato."""
    try:
        stock_data_prophet = yf.download(ticker, start='1990-01-01', end=datetime.now().strftime('%Y-%m-%d'))
        if stock_data_prophet.empty:
            raise ValueError("Dati storici non disponibili per il ticker fornito.")
        stock_data_prophet.reset_index(inplace=True)
        return stock_data_prophet
    except Exception as e:
        print(f"Errore durante il recupero dei dati: {e}")
        return pd.DataFrame()

def preprocess_data_prophet(data_prophet):
    """Preprocessa i dati finanziari per Prophet."""
    if data_prophet.empty:
        raise ValueError("Nessun dato storico disponibile per il preprocessing.")
    data_prophet['Date'] = pd.to_datetime(data_prophet['Date'])
    data_prophet = data_prophet[['Date', 'Close']]
    data_prophet.columns = ['ds', 'y']
    return data_prophet

def get_economic_data_prophet():
    """Scarica dati economici reali."""
    start_date_prophet = '1990-01-01'
    end_date_prophet = datetime.now().strftime('%Y-%m-%d')

    # Scarica dati economici da FRED tramite pandas_datareader
    unemployment_rate_prophet = pdr.get_data_fred('UNRATE', start=start_date_prophet, end=end_date_prophet)
    inflation_rate_prophet = pdr.get_data_fred('CPIAUCSL', start=start_date_prophet, end=end_date_prophet)
    interest_rate_prophet = pdr.get_data_fred('FEDFUNDS', start=start_date_prophet, end=end_date_prophet)

    # Prepara il DataFrame unendo i dati economici su base temporale
    economic_data_prophet = pd.DataFrame({
        'ds': unemployment_rate_prophet.index,
        'unemployment_rate': unemployment_rate_prophet['UNRATE'],
        'inflation_rate': inflation_rate_prophet['CPIAUCSL'].pct_change() * 100,  # Tasso di crescita annuale della CPI
        'interest_rate': interest_rate_prophet['FEDFUNDS']
    })

    # Riempie i NaN
    economic_data_prophet = economic_data_prophet.fillna(method='bfill').fillna(method='ffill')

    return economic_data_prophet

def analizza_fasi_prophet(ticker):
    """Analizza le fasi del ciclo di vita dell'azienda in base ai flussi di cassa."""
    azienda_prophet = yf.Ticker(ticker)
    flussi_cassa_prophet = azienda_prophet.cashflow
    flussi_cassa_prophet = flussi_cassa_prophet.T  # Trasponi per avere le date come righe

    flussi_cassa_prophet.columns = flussi_cassa_prophet.columns.str.strip()

    try:
        flussi_cassa_prophet = flussi_cassa_prophet[['Operating Cash Flow', 'Investing Cash Flow', 'Financing Cash Flow']]
    except KeyError as e:
        print(f"Errore: {e}. Verifica i nomi delle colonne disponibili.")
        return None

    flussi_cassa_prophet.sort_index(ascending=False, inplace=True)
    last_3_years = flussi_cassa_prophet.head(3)

    fasi_prophet = pd.DataFrame(columns=['ds', 'fase'])

    for index, row in last_3_years.iterrows():
        operativo_prophet = row['Operating Cash Flow']
        investimento_prophet = row['Investing Cash Flow']
        finanziamento_prophet = row['Financing Cash Flow']

        if operativo_prophet > 0 and investimento_prophet < 0 and finanziamento_prophet > 0:
            fase_prophet = 2  # Crescita
        elif operativo_prophet < 0 and investimento_prophet < 0 and finanziamento_prophet > 0:
            fase_prophet = 1  # Introduzione
        elif operativo_prophet > 0 and investimento_prophet < 0 and finanziamento_prophet < 0:
            fase_prophet = 3  # Maturità
        elif (operativo_prophet < 0 and investimento_prophet < 0 and finanziamento_prophet < 0) or \
             (operativo_prophet > 0 and investimento_prophet > 0 and finanziamento_prophet > 0) or \
             (operativo_prophet > 0 and investimento_prophet > 0 and finanziamento_prophet < 0):
            fase_prophet = 4  # Shake-out
        elif operativo_prophet < 0 and investimento_prophet > 0 and finanziamento_prophet < 0:
            fase_prophet = 5  # Declino
        else:
            fase_prophet = 0  # N/A

        year_prophet = index.year
        date_range_prophet = pd.date_range(start=f"{year_prophet}-01-01", end=f"{year_prophet}-12-31", freq='D')

        df_temp_prophet = pd.DataFrame({'ds': date_range_prophet, 'fase': fase_prophet})

        fasi_prophet = pd.concat([fasi_prophet, df_temp_prophet], ignore_index=True)

    fasi_prophet['ds'] = pd.to_datetime(fasi_prophet['ds'])
    fasi_prophet.sort_values(by='ds', inplace=True)

    today_prophet = pd.to_datetime("today").normalize()
    all_dates_prophet = pd.date_range(start=fasi_prophet['ds'].min(), end=today_prophet, freq='D')

    fasi_prophet = fasi_prophet.set_index('ds').reindex(all_dates_prophet, method='ffill').reset_index()
    fasi_prophet.columns = ['ds', 'fase']

    return fasi_prophet

def fit_prophet_model(data_prophet, exogenous_data_prophet, lifecycle_data_prophet):
    """Addestra il modello Prophet con variabili esogene inclusa la fase del ciclo di vita."""
    model_prophet = Prophet(yearly_seasonality=True, daily_seasonality=False,changepoint_prior_scale=0.0000000000000000005)

    for col in exogenous_data_prophet.columns:
        if col != 'ds':
            model_prophet.add_regressor(col)

    # Aggiungi la fase del ciclo di vita come regressore
    exogenous_data_prophet = pd.merge(exogenous_data_prophet, lifecycle_data_prophet, on='ds', how='left')
    exogenous_data_prophet = exogenous_data_prophet.fillna(method='ffill')

    full_data_prophet = pd.merge(data_prophet, exogenous_data_prophet, on='ds')
    model_prophet.fit(full_data_prophet)
    return model_prophet

def predict_future_prophet(model_prophet, data_prophet, exogenous_data_prophet, lifecycle_data_prophet, end_year=2040):
    """Prevede i dati futuri usando il modello Prophet."""
    future_prophet = model_prophet.make_future_dataframe(periods=(pd.date_range(start=data_prophet['ds'].max() + pd.Timedelta(days=1), end=f'{end_year}-12-31').size))

    future_prophet = pd.merge(future_prophet, exogenous_data_prophet, on='ds', how='left')
    future_prophet = pd.merge(future_prophet, lifecycle_data_prophet, on='ds', how='left')
    future_prophet = future_prophet.fillna(method='bfill').fillna(method='ffill')

    forecast_prophet = model_prophet.predict(future_prophet)
    return forecast_prophet



def main():
    # Aggiungi il logo centrato all'inizio dell'app
    logo = Image.open("Logo.png")  # Carica l'immagine
    # Crea tre colonne e centra il logo nella colonna centrale
    col1, col2, col3 = st.columns([1, 2, 1])  # Proporzioni delle colonne
    with col2:  # Colonna centrale
        st.image(logo, width=300, use_column_width=False)  # Mostra l'immagine

    st.title("Firm Life Cycle and Stock Return: a predictive analysis")

    # Aggiungi l'avviso per la versione beta e disclaimer
    st.warning(
        "⚠️ **Versione Beta**: Questo strumento è in fase sperimentale. Le previsioni potrebbero essere imprecise e non devono essere considerate come consigli finanziari.")
    macro_trend = False
    visualdata = False
    # Input per il ticker
    ticker = st.text_input("Inserisci il ticker dell'azienda:", value="AAPL")
    if st.checkbox("Mostra i dati"):
        visualdata = True
    if st.checkbox("Mostra Trend di Lungo Periodo"):
        macro_trend = True
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
            scaled_economic, scaler_economic = preprocess_economic_data(
                combined_data[['Unemployment', 'Inflation', 'Interest']])

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
            model.fit(X, y, epochs=10, batch_size=32, verbose=1)

            # Previsione futura
            future_end_date = datetime(2030, 1, 1)
            future_days = (future_end_date - combined_data.index[-1]).days
            future_predictions = predict_future(model, scaled_price, scaled_economic, scaled_phase, time_step,
                                                future_days)

            future_dates = [combined_data.index[-1] + timedelta(days=i) for i in range(1, future_days + 1)]
            future_predictions = scaler_price.inverse_transform(np.array(future_predictions).reshape(-1, 1))

            # Combina i prezzi storici e i prezzi predetti in un'unica serie
            all_prices = np.concatenate((combined_data['Close'].values, future_predictions.flatten()))
            all_dates = np.concatenate((combined_data.index.values, future_dates))
            data_prophet = get_financial_data_prophet(ticker)
            if macro_trend == True:

                if data_prophet.empty:
                    print("Impossibile procedere senza dati storici.")
                    return

                price_data_prophet = preprocess_data_prophet(data_prophet)

                # Scarica e preprocessa i dati economici reali
                economic_data_prophet = get_economic_data_prophet()

                # Analizza le fasi del ciclo di vita dell'azienda
                lifecycle_data_prophet = analizza_fasi_prophet(ticker)
                if lifecycle_data_prophet is None:
                    print("Impossibile procedere senza i dati delle fasi del ciclo di vita.")
                    return

                # Addestra il modello con variabili esogene e ciclo di vita
                model_prophet = fit_prophet_model(price_data_prophet, economic_data_prophet, lifecycle_data_prophet)

                # Prevedi i dati futuri
                forecast_prophet = predict_future_prophet(model_prophet, price_data_prophet, economic_data_prophet,
                                                      lifecycle_data_prophet)



            # Grafico interattivo con Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=combined_data.index, y=combined_data['Close'], mode='lines', name='Dati Storici',
                                     line=dict(color='blue')))
            if macro_trend == True:
                fig.add_trace(go.Scatter(x=all_dates, y=forecast_prophet['yhat'], mode='lines', name='Macro Trend',
                                     line=dict(color='green')))
            fig.add_trace(go.Scatter(x=all_dates, y=all_prices, mode='lines', name='Previsioni', line=dict(color='red')))

            fig.update_layout(title=f'Previsione del Prezzo delle Azioni per {ticker}',
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
                st.write(
                    f"L'azienda si trova nella fase **{current_phase_num_print}** ({current_phase_desc}) del suo ciclo di vita. {current_phase_desc_esplit}")
            else:
                st.write("Non è stato possibile determinare la fase attuale dell'azienda.")
    st.write("")  # Riga vuota
    st.write("")  # Riga vuota
    st.write("")  # Riga vuota
    st.write("")  # Riga vuota
    st.write("")  # Riga vuota
    # Aggiungi un container per il secondo logo e il box di testo
    with st.container():

        # Aggiungi un box di testo
        st.subheader("Informazioni sul progetto")
        st.write(
            "Questo applicativo è il frutto di un'approfondita ricerca condotta nell'ambito della tesi di laurea magistrale da uno studente dell'Università degli studi del Piemonte Orientale.")

        # Aggiungi il secondo logo centrato
        logo2 = Image.open("Logo2.png")  # Carica il secondo logo
        col1, col2, col3 = st.columns([1, 2, 1])  # Proporzioni delle colonne
        with col2:  # Colonna centrale
            st.image(logo2, width=250, use_column_width=False)  # Mostra il secondo logo


if __name__ == "__main__":
    main()
