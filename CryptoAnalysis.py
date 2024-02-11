import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
from sqlalchemy import create_engine
import dash_bootstrap_components as dbc
import dash_table
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.signal import periodogram  
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from scipy import signal
from statsmodels.tsa.api import ExponentialSmoothing
from dash.exceptions import PreventUpdate  
from dash.dependencies import Input, Output, State
from statsmodels.tsa.api import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from abc import ABCMeta, abstractmethod
from concurrent.futures import ProcessPoolExecutor



import warnings
warnings.filterwarnings('ignore')


# Conexión a la base de datos
engine = create_engine("sqlite:///C:\\Users\\Usuario\\OneDrive - Universidad Estatal a Distancia\\Promidat\\9- Proyecto Final\\cripto_analisis.db", echo=False)

# Inicilización de la aplicación
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])



# Cache para almacenar los datos de las tablas
cache = {}



# Funciones para el modelado y predicción
def fit_model(model, order, seasonal_order, ts, test):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modelo = model(ts, order=order, seasonal_order=seasonal_order)
            model_fit = modelo.fit(disp=False)
            pred = model_fit.forecast(len(test))
        return pred
    except Exception as e:
        return None

# Función para el modelado y predicción en paralelo
def fit_parallel(model, order_range, seasonal_order, ts, test):
    results = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(fit_model, model, order, seasonal_order, ts, test) for order in order_range]
        results = [future.result() for future in futures if future.result() is not None]
    return results

class BaseModelo(metaclass = ABCMeta):    
  @abstractmethod
  def fit(self):
    pass

class Modelo(BaseModelo):
  def __init__(self, ts):
    self.__ts = ts
    self._coef = None
  
  @property
  def ts(self):
    return self.__ts  
  
  @ts.setter
  def ts(self, ts):
    if(isinstance(ts, pd.core.series.Series)):
      if(ts.index.freqstr != None):
        self.__ts = ts
      else:
        warnings.warn('ERROR: No se indica la frecuencia de la serie de tiempo.')
    else:
      warnings.warn('ERROR: El parámetro ts no es una instancia de serie de tiempo.')
  
  @property
  def coef(self):
    return self._coef

class BasePrediccion(metaclass = ABCMeta):    
  @abstractmethod
  def forecast(self):
    pass

class Prediccion(BasePrediccion):
  def __init__(self, modelo):
    self.__modelo = modelo
  
  @property
  def modelo(self):
    return self.__modelo  
  
  @modelo.setter
  def modelo(self, modelo):
    if(isinstance(modelo, Modelo)):
      self.__modelo = modelo
    else:
      warnings.warn('El objeto debe ser una instancia de Modelo.')
  
class meanfPrediccion(Prediccion):
  def __init__(self, modelo):
    super().__init__(modelo)
  
  def forecast(self, steps = 1):
    res = []
    for i in range(steps):
      res.append(self.modelo.coef)
    
    start = self.modelo.ts.index[-1]
    freq  = self.modelo.ts.index.freqstr
    fechas = pd.date_range(start = start, periods = steps+1, freq = freq)
    fechas = fechas.delete(0)
    res = pd.Series(res, index = fechas)
    return(res)

class LSTM_TSPrediccion(Prediccion):
  def __init__(self, modelo):
    super().__init__(modelo)
    self.__scaler = MinMaxScaler(feature_range = (0, 1))
    self.__X = self.__scaler.fit_transform(self.modelo.ts.to_frame())
  
  def __split_sequence(self, sequence, n_steps):
    X, y = [], []
    for i in range(n_steps, len(sequence)):
      X.append(self.__X[i-n_steps:i, 0])
      y.append(self.__X[i, 0])
    return np.array(X), np.array(y)
  
  def forecast(self, steps = 1):
    res = []
    p = self.modelo.p
    for i in range(steps):
      y_pred = [self.__X[-p:].tolist()]
      X, y = self.__split_sequence(self.__X, p)
      X = np.reshape(X, (X.shape[0], X.shape[1], 1))
      self.modelo.m.fit(X, y, epochs = 10, batch_size = 1, verbose = 0)
      pred = self.modelo.m.predict(y_pred)
      res.append(self.__scaler.inverse_transform(pred).tolist()[0][0])
      self.__X = np.append(self.__X, pred.tolist(), axis = 0)
    
    start  = self.modelo.ts.index[-1]
    freq   = self.modelo.ts.index.freqstr
    fechas = pd.date_range(start = start, periods = steps+1, freq = freq)
    fechas = fechas.delete(0)
    res = pd.Series(res, index = fechas)
    return(res)

class LSTM_TS(Modelo):
  def __init__(self, ts, p = 1, lstm_units = 50, dense_units = 1, optimizer = 'rmsprop', loss = 'mse'):
    super().__init__(ts)
    self.__p = p
    self.__m = Sequential()
    self.__m.add(LSTM(units = lstm_units, input_shape = (p, 1)))
    self.__m.add(Dense(units = dense_units))
    self.__m.compile(optimizer = optimizer, loss = loss)
  
  @property
  def m(self):
    return self.__m
  
  @property
  def p(self):
    return self.__p
  
  def fit(self):
    res = LSTM_TSPrediccion(self)
    return(res)

class ARIMA_Prediccion(Prediccion):
  def __init__(self, modelo, p, d, q):
    super().__init__(modelo)
    self.__p = p
    self.__d = d
    self.__q = q
  
  @property
  def p(self):
    return self.__p
  
  @property
  def d(self):
    return self.__d
  
  @property
  def q(self):
    return self.__q
  
  def forecast(self, steps = 1):
    res = self.modelo.forecast(steps)
    return(res)

class SARIMA_Prediccion(Prediccion):
    def __init__(self, modelo, p, d, q, P, D, Q, S):
        super().__init__(modelo)
        self.__p = p
        self.__d = d
        self.__q = q
        self.__P = P
        self.__D = D
        self.__Q = Q
        self.__S = S
  
    @property
    def p(self):
        return self.__p
  
    @property
    def d(self):
        return self.__d
  
    @property
    def q(self):
        return self.__q
  
    @property
    def P(self):
        return self.__P
  
    @property
    def D(self):
        return self.__D
  
    @property
    def Q(self):
        return self.__Q
  
    @property
    def S(self):
        return self.__S
  
    def forecast(self, steps=1):
        res = self.modelo.forecast(steps)
        return res



class ARIMA_calibrado(Modelo):
  def __init__(self, ts, test):
    super().__init__(ts)
    self.__test = test
  
  @property
  def test(self):
    return self.__test  
  
  @test.setter
  def test(self, test):
    if(isinstance(test, pd.core.series.Series)):
      if(test.index.freqstr != None):
        self.__test = test
      else:
        warnings.warn('ERROR: No se indica la frecuencia de la serie de tiempo.')
    else:
      warnings.warn('ERROR: El parámetro ts no es una instancia de serie de tiempo.')
  
    def fit(self, ar=[0, 1, 2]):
        # Insert new code for parallel processing
        order_range = [(p, d, q) for p in ar for d in ar for q in ar]
        results = fit_parallel(SARIMAX, order_range, [0, 0, 0, 0])
        best_result = min(results, key=lambda x: sum((x - self.test) ** 2))
        res, res_p, res_d, res_q = best_result, best_result.order[0], best_result.order[1], best_result.order[2]
        return ARIMA_Prediccion(res, res_p, res_d, res_q)

class SARIMA_calibrado(Modelo):
  def __init__(self, ts, test):
    super().__init__(ts)
    self.__test = test
  
  @property
  def test(self):
    return self.__test  
  
  @test.setter
  def test(self, test):
    if(isinstance(test, pd.core.series.Series)):
      if(test.index.freqstr != None):
        self.__test = test
      else:
        warnings.warn('ERROR: No se indica la frecuencia de la serie de tiempo.')
    else:
      warnings.warn('ERROR: El parámetro ts no es una instancia de serie de tiempo.')
  
  def fit(self, ar = [0, 1, 2], es = [0, 1], S = None):
    if S is None:
      warnings.warn('ERROR: No se indica el periodo a utilizar (S).')
      return(None)
    res, res_p, res_d, res_q, res_P, res_D, res_Q = (None, 0, 0, 0, 0, 0, 0)
    error = float("inf")
    for p in ar:
      for d in ar:
        for q in ar:
          for P in es:
            for D in es:
              for Q in es:
                try:
                  with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    modelo    = SARIMAX(self.ts, order = [p, d, q], seasonal_order = [P, D, Q, S])
                    model_fit = modelo.fit(disp = False)
                  pred      = model_fit.forecast(len(self.test))
                  mse       = sum((pred - self.test)**2)
                  if mse < error:
                    res_p = p
                    res_d = d
                    res_q = q
                    res_P = P
                    res_D = D
                    res_Q = Q
                    error = mse
                    res = model_fit
                except:
                  modelo = None
    return(SARIMA_Prediccion(res, res_p, res_d, res_q, res_P, res_D, res_Q, S))



class HW_Prediccion:
    def __init__(self, modelo, alpha, beta, gamma):
        self.__modelo = modelo
        self.__alpha = alpha
        self.__beta = beta
        self.__gamma = gamma

    @property
    def alpha(self):
        return self.__alpha

    @property
    def beta(self):
        return self.__beta

    @property
    def gamma(self):
        return self.__gamma

    def forecast(self, steps=1):
        res = self.__modelo.forecast(steps)
        return res

class HW_calibrado:
    def __init__(self, ts, test, trend='add', seasonal='add'):
        self.__ts = ts
        self.__test = test
        self.__modelo = ExponentialSmoothing(ts, trend=trend, seasonal=seasonal)

    @property
    def test(self):
        return self.__test

    @test.setter
    def test(self, test):
        if isinstance(test, pd.core.series.Series):
            if test.index.freqstr is not None:
                self.__test = test
            else:
                warnings.warn('ERROR: No se indica la frecuencia de la serie de tiempo.')
        else:
            warnings.warn('ERROR: El parámetro ts no es una instancia de serie de tiempo.')

    def fit(self, paso=0.1):
        error = float("inf")
        alpha = paso
        while alpha <= 1:
            beta = 0
            while beta <= 1:
                gamma = 0
                while gamma <= 1:
                    model_fit = self.__modelo.fit(smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma)
                    pred = model_fit.forecast(len(self.__test))
                    mse = sum((pred - self.__test) ** 2)
                    if mse < error:
                        res_alpha = alpha
                        res_beta = beta
                        res_gamma = gamma
                        error = mse
                        res = model_fit
                    gamma += paso
                beta += paso
            alpha += paso
        return HW_Prediccion(res, res_alpha, res_beta, res_gamma)

# Definición de la clase Periodograma
class Periodograma:
    def __init__(self, ts):
        self.__ts = ts
        self.__freq, self.__spec = signal.periodogram(ts)

    @property
    def ts(self):
        return self.__ts

    @property
    def freq(self):
        return self.__freq

    @property
    def spec(self):
        return self.__spec

    def mejor_freq(self, best=3):
        res = np.argsort(-self.spec)
        res = res[res != 1][0:best]
        return self.freq[res]

    def mejor_periodos(self, best=3):
        return 1 / self.mejor_freq(best)

    def plot_periodograma(self, best=3):
        res = self.mejor_freq(best)
        plt.plot(self.freq, self.spec, color="darkgray")
        for i in range(best):
            plt.axvline(x=res[i], label="Mejor " + str(i + 1), ls='--', c=np.random.rand(3))
        plt.legend(loc="best")
        plt.show()

    def plotly_periodograma(self, best=3):
        res = self.mejor_freq(best)
        fig = go.Figure()
        no_plot = fig.add_trace(
            go.Scatter(x=self.freq, y=self.spec,
                       mode='lines+markers', line_color='darkgray')
        )
        for i in range(best):
            v = np.random.rand(3)
            color = "rgb(" + str(v[0]) + ", " + str(v[1]) + ", " + str(v[2]) + ")"
            fig.add_vline(x=res[i], line_width=2, line_dash="dash",
                          annotation_text="Mejor " + str(i + 1),
                          line_color=color)

        return fig

#Clase ARIMA  
class ARIMA_Prediccion:
  def __init__(self, modelo, p, d, q):
    super().__init__(modelo)
    self.__p = p
    self.__d = d
    self.__q = q
  
  @property
  def p(self):
    return self.__p
  
  @property
  def d(self):
    return self.__d
  
  @property
  def q(self):
    return self.__q
  
  def forecast(self, steps = 1):
    res = self.modelo.forecast(steps)
    return(res)

#Clase SARIMA
class SARIMA_Prediccion:
  def __init__(self, modelo, p, d, q, P, D, Q, S):
    super().__init__(modelo)
    self.__p = p
    self.__d = d
    self.__q = q
    self.__P = P
    self.__D = D
    self.__Q = Q
    self.__S = S
  
  @property
  def p(self):
    return self.__p
  
  @property
  def d(self):
    return self.__d
  
  @property
  def q(self):
    return self.__q
  
  @property
  def P(self):
    return self.__P
  
  @property
  def D(self):
    return self.__D
  
  @property
  def Q(self):
    return self.__Q
  
  @property
  def S(self):
    return self.__S
  
  def forecast(self, steps = 1):
    res = self.modelo.forecast(steps)
    return(res)

# Layout de la aplicación
app.layout = dbc.Container([
    dcc.Tabs([
        dcc.Tab(label='Vista Tabla de Datos', children=[
            html.H1("Visor de Tabla: ", className="mt-3 mb-4"),

            dbc.Row([
                # Dropdown for selecting the table
                dbc.Col(html.Label("Seleccione Tabla:")),
                dbc.Col(
                    dcc.Dropdown(
                        id='table-dropdown',
                        options=[
                            {'label': 'BTC', 'value': 'BTC'},
                            {'label': 'ETH', 'value': 'ETH'},
                            {'label': 'BNB', 'value': 'BNB'}
                        ],
                        value='BTC'  # Valor seleccionado por defecto
                    )
                ),
            ], className="mb-4"),

            # Muestra la tabla seleccionada
            dbc.Row([
                dbc.Col(dash_table.DataTable(id='table-content')),
            ]),
        ]),
        dcc.Tab(label='Análisis de la Serie', children=[
            html.H1("Visor de Gráficos", className="mt-3 mb-4"),

            # Nuevos componentes para el gráfico de la serie
            html.H3(id='graph-title', className="mt-3 mb-2"),
            dcc.Graph(id='time-series-graph'),

            # Componentes adicionales para el gráfico de la descomposición de la serie
            html.H3("Descomposición de la serie", className="mt-3 mb-2"),
            dcc.Graph(id='seasonal-decomposition-graph'),

            # Componentes adicionales para el gráfico del periodograma
            html.H3("Periodograma", className="mt-3 mb-2"),
            dcc.Input(id='periodogram-input', type='number', placeholder='Enter periodogram parameter', value=3),
            dcc.Graph(id='periodogram-graph'),
        ]),
         dcc.Tab(label='Modelado y Predicción', children=[
            html.H1("Visor de Gráficos", className="mt-3 mb-4"),
             # Dropdowns para seleccionar el número de días para entrenamiento y prueba
            dbc.Button('Generar Pronóstico', id='forecast-button', color='primary', className='mt-3 mb-4'),


                  # Dropdown para seleccionar el modelo de pronóstico
            dcc.Dropdown(
            id='forecast-model-dropdown',
            options=[
                {'label': 'Holt-Winters', 'value': 'Holt-Winters'},
                {'label': 'Holt-Winters Calibrado', 'value': 'Holt-Winters Calibrado'},
                {'label': 'ARIMA', 'value': 'ARIMA'},
                {'label': 'ARIMA Calibrado', 'value': 'ARIMA Calibrado'},
                {'label': 'Deep Learning', 'value': 'Deep Learning'}
            ],
            value='hw'  # Valor seleccionado por defecto
            ),


            # Agregar el componente de entrada para el número de días
            dcc.Input(id='forecast-days-input', type='number', placeholder='Enter the number of days for training and testing', value=30),

            html.H3("Gráfico de Predicciones", className="mt-3 mb-2"),
            # Gráfico para mostrar el resultado del pronóstico
            dcc.Graph(id='forecast-graph'),
            ]),

            dcc.Tab(label='Predicción Final', children=[
            html.H1("Visor de Gráficos", className="mt-3 mb-4"),

             # Dropdowns para seleccionar el número de días para entrenamiento y prueba
            dbc.Button('Generar Pronóstico', id='forecast-button2', color='primary', className='mt-3 mb-4'),
            
            # Add a new button for exporting Excel
            dbc.Button('Exportar', id='export-button2', color='success', className='mt-3 mb-4'),


        # Dropdown para seleccionar el modelo de pronóstico
            dcc.Dropdown(
            id='forecast-model-dropdown2',
            options=[
                {'label': 'Holt-Winters', 'value': 'Holt-Winters'},
                {'label': 'Holt-Winters Calibrado', 'value': 'Holt-Winters Calibrado'},
                {'label': 'ARIMA', 'value': 'ARIMA'},
                {'label': 'ARIMA Calibrado', 'value': 'ARIMA Calibrado'},
                {'label': 'Deep Learning', 'value': 'Deep Learning'}
            ],
            value='hw'   # Valor seleccionado por defecto
            ),
    # Agregar el componente de entrada para el número de días
            dcc.Input(id='forecast-days-input2', type='number', placeholder='Enter the number of days for training and testing', value=30),
            

            html.H3("Gráfico de Predicciones", className="mt-3 mb-2"),
            # Grafigo para mostrar el resultado del pronóstico
            dcc.Graph(id='forecast-graph2'),

            ]),
    ]),
])

# Callback para actualizar la tabla de datos
@app.callback(
    Output('table-content', 'data'),
    [Input('table-dropdown', 'value')]
)


def update_table(selected_table):
    # Verificar si los datos ya están en caché
    if selected_table in cache:
        df = cache[selected_table]
    else:
        # Leer la tabla seleccionada de la base de datos
        df = pd.read_sql(selected_table, con=engine)

        # Almacenar los datos en caché para uso futuro
        cache[selected_table] = df

    # formatear la informacion como dash_table.DataTable
    data = df.to_dict('records')

    return data

# Callback para actualizar el gráfico de la serie de tiempo
@app.callback(
    [Output('time-series-graph', 'figure'),
     Output('graph-title', 'children'),
     Output('seasonal-decomposition-graph', 'figure'),
     Output('periodogram-graph', 'figure')],
    [Input('table-dropdown', 'value'),
     Input('periodogram-input', 'value')]
)
def update_time_series_graph(selected_table, periodogram_param):
    # Verificar si los datos ya están en caché
    if selected_table in cache:
        df = cache[selected_table]
    else:
        # Leer la tabla seleccionada de la base de datos
        df = pd.read_sql(selected_table, con=engine)

        # Almacenar los datos en caché para uso futuro
        cache[selected_table] = df

    # Crear una serie de Pandas para la serie de tiempo
    df_ts = pd.Series(df['Close'].values, index = pd.date_range(start = '2020-01-01', periods = len(df), freq = 'D'))

    # Calcula la media y la desviación estándar de la serie de tiempo
    mean, std = stats.norm.fit(np.diff(df_ts))

    # Crea un histograma usando el estilo de Seaborn
    hist_data = go.Histogram(x=np.diff(df_ts), nbinsx=25, histnorm='probability density', opacity=0.6,
                             marker=dict(color='black'))

    # Crear un gráfico de densidad usando el estilo Seaborn
    kde_data = go.Scatter(x=np.linspace(min(np.diff(df_ts)), max(np.diff(df['Close'])), 100),
                          y=stats.norm.pdf(np.linspace(min(np.diff(df_ts)), max(np.diff(df_ts)), 100),
                                          mean, std),
                          mode='lines', line=dict(color='palegreen', width=3), name='Densidad')

    # Crear un gráfico de densidad usando el estilo Seaborn
    density_data = go.Scatter(x=np.linspace(min(np.diff(df_ts)), max(np.diff(df_ts)), 100),
                             y=stats.norm.pdf(np.linspace(min(np.diff(df_ts)), max(np.diff(df_ts)), 100),
                                             mean, std),
                             mode='lines', line=dict(color='pink', width=3), name='Normalidad')

    layout = go.Layout(title=f'Normalización Serie - {selected_table}', title_x=0.5,xaxis=dict(title='Diferencia'),
                       yaxis=dict(title='Densidad'))

    # Convertir los datos en un objeto de figura de Plotly
    time_series_fig = go.Figure(data=[hist_data, kde_data, density_data], layout=layout)

   

    # Proceda con la descomposición estacional
    r = seasonal_decompose(df_ts, model='additive', period=1)  

    # Crear un gráfico de subtramas con 4 filas y 1 columna
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'))

    # Agregar las trazas para cada una de las subtramas
    fig.add_trace(go.Scatter(x=df_ts.index, y=r.observed, mode='lines', name='Original'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ts.index, y=r.trend, mode='lines', name='Trend', line=dict(dash='solid')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_ts.index, y=r.seasonal, mode='lines', name='Seasonal', line=dict(dash='dot')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_ts.index, y=r.resid, mode='lines', name='Residual', line=dict(dash='dashdot')), row=4, col=1)

    fig.update_layout(title=f'Descomposición de la Serie - {selected_table}',title_x=0.5 ,xaxis=dict(title=''))

    # Use la figura de Plotly para crear una figura de Plotly Express
    seasonal_decomposition_fig = fig

    # Crear un objeto Periodograma
    periodogram_obj = Periodograma(np.diff(df['Close']))

    # Usar el objeto Periodograma para crear un gráfico de Plotly
    periodogram_fig = periodogram_obj.plotly_periodograma(best=periodogram_param)

    # Setear el título del gráfico
    periodogram_fig.update_layout(title=f'Periodograma - {selected_table}', title_x=0.5)

    return time_series_fig, f'Normalización Serie', seasonal_decomposition_fig, periodogram_fig



# Callback para actualizar el gráfico de pronóstico
@app.callback(
    [Output('forecast-graph', 'figure')],
    [Input('forecast-button', 'n_clicks')],
    [State('table-dropdown', 'value'),
     State('forecast-days-input', 'value'),
     State('forecast-model-dropdown', 'value')]  
)

def update_forecast_graph(n_clicks, selected_table, forecast_days,selected_model):
    if n_clicks is None or n_clicks == 0 or selected_table is None:
        raise PreventUpdate

    # Verificar si los datos ya están en caché
    if selected_table in cache:
        df = cache[selected_table]
    else:
        # Leer la tabla seleccionada de la base de datos
        df = pd.read_sql(selected_table, con=engine)

        # Almacenar los datos en caché para uso futuro
        cache[selected_table] = df

    # Crear una serie de Pandas para la serie de tiempo
    df_ts = pd.Series(df['Close'].values, index=pd.date_range(start='2020-01-01', periods=len(df), freq='D'))

    # Usar el valor de entrada para el número de días para entrenamiento y prueba
    df_train = df_ts[:len(df_ts) - forecast_days]
    df_test = df_ts[len(df_ts) - forecast_days:]

     # Código existente para ExponentialSmoothing y pronóstico
    if selected_model == 'Holt-Winters':
        modelo = ExponentialSmoothing(df_train, trend='add', seasonal='add')
        modelo_fit = modelo.fit()
        pred = modelo_fit.forecast(forecast_days)
    elif selected_model == 'Holt-Winters Calibrado':
        modelo_calibrado = HW_calibrado(df_train, df_test)
        modelo_calibrado_fit = modelo_calibrado.fit(0.1)
        pred = modelo_calibrado_fit.forecast(forecast_days)
    elif selected_model == 'ARIMA':
        # ARIMA
        modelo = SARIMAX(df_train, order=(1, 1, 1), seasonal_order=(1, 0, 1, 12))
        modelo_fit = modelo.fit(S=12)
        pred = modelo_fit.forecast(forecast_days)
    elif selected_model == 'ARIMA Calibrado':
        modelo_calibrado     = SARIMA_calibrado(df_train, df_test)
        modelo_calibrado_fit = modelo_calibrado.fit(S = 5)
        pred = modelo_calibrado_fit.forecast(forecast_days)
    elif selected_model == 'Deep Learning':
        # LSTM
        modelo = LSTM_TS(df_train, 67)
        modelo_fit = modelo.fit()
        pred = modelo_fit.forecast(forecast_days)

    series = [df_train, df_test, pred]
    nombres = ["Entrenamiento", "Prueba", selected_model]

    # Crear la figura para el pronóstico de Holt-Winters
    fig = go.Figure()
    for i in range(len(series)):
        no_plot = fig.add_trace(
            go.Scatter(x=series[i].index.tolist(), y=series[i].values.tolist(),
                       mode='lines+markers', name=nombres[i])
        )

    fig.update_layout(title=f'Múltiples Predicciones - {selected_table}', title_x=0.5, xaxis=dict(title='Fecha'), yaxis=dict(title='Precio'))

    no_plot = fig.update_xaxes(rangeslider_visible=True)

    return fig,  


# Callback para actualizar el gráfico de pronóstico
@app.callback(
    [Output('forecast-graph2', 'figure')],
    [Input('forecast-button2', 'n_clicks')],
    [State('table-dropdown', 'value'),
     State('forecast-days-input2', 'value'),
     State('forecast-model-dropdown2', 'value')] 
)

def update_forecast_graph2(n_clicks, selected_table, forecast_days,selected_model):
    if n_clicks is None or n_clicks == 0 or selected_table is None:
        raise PreventUpdate

    #  Verificar si los datos ya están en caché
    if selected_table in cache:
        df = cache[selected_table]
    else:
        # Leer la tabla seleccionada de la base de datos
        df = pd.read_sql(selected_table, con=engine)

        # Almacenar los datos en caché para uso futuro
        cache[selected_table] = df

    # Crear una serie de Pandas para la serie de tiempo
    df_ts = pd.Series(df['Close'].values, index=pd.date_range(start='2020-01-01', periods=len(df), freq='D'))

    # Usar el valor de entrada para el número de días para entrenamiento y prueba
    df_train = df_ts[:len(df_ts) - forecast_days]
    df_test = df_ts[len(df_ts) - forecast_days:]

     # Código existente para ExponentialSmoothing y pronóstico
    if selected_model == 'Holt-Winters':
        modelo = ExponentialSmoothing(df_ts, trend='add', seasonal='add')
        modelo_fit = modelo.fit()
        pred = modelo_fit.forecast(forecast_days)
    elif selected_model == 'Holt-Winters Calibrado':
        metodo_hw = ExponentialSmoothing(df_ts, trend = 'add', seasonal = 'add')
        pred   = metodo_hw.fit(smoothing_level = 0.1, smoothing_slope = 0, smoothing_seasonal = 0.4).forecast(30)  
    elif selected_model == 'ARIMA':
        # ARIMA
        modelo = SARIMAX(df_ts, order=(1, 1, 1), seasonal_order=(1, 0, 1, 30))
        modelo_fit = modelo.fit(S=30)
        pred = modelo_fit.forecast(forecast_days)
    elif selected_model == 'ARIMA Calibrado':
        modelo_calibrado     = SARIMA_calibrado(df_ts, df_ts)
        modelo_calibrado_fit = modelo_calibrado.fit(S = 5)
        pred = modelo_calibrado_fit.forecast(forecast_days)
    elif selected_model == 'Deep Learning':
        # LSTM
        modelo = LSTM_TS(df_ts, 67)
        modelo_fit = modelo.fit()
        pred = modelo_fit.forecast(forecast_days)

    series = [df_train, df_test, pred]
    nombres = ["Entrenamiento", "Prueba", selected_model]

    # Crear la figura para los pronósticos
    fig = go.Figure()
    no_plot = fig.add_trace(
            go.Scatter(x=df_ts.index.tolist(), y=df_ts.values.tolist(),
                       mode='lines+markers', name="Original")
        )
    no_plot = fig.add_trace(
            go.Scatter(x = pred.index.tolist(), y = pred.values.tolist(), 
                        mode = 'lines+markers', name = "Predicción")
        )

    fig.update_layout(title=f'Múltiples Predicciones - {selected_table}', title_x=0.5, xaxis=dict(title='Fecha'), yaxis=dict(title='Precio'))

    no_plot = fig.update_xaxes(rangeslider_visible=True)

    return fig, 

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
