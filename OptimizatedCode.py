import math
import boto3
import pandas as pd
import numpy as np
import pathlib
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from io import StringIO, BytesIO
from datetime import datetime, timedelta

class Extract:

    ## Método constructor de la clase:
    def __init__(self, typeService, xetraBucket, arg_date):
        self.typeService = boto3.resource(typeService)
        self.xetraBucket = self.typeService.Bucket(xetraBucket)
        self.arg_date = arg_date

    ## Método encargado de extraer los objetos de la base de datos "Xetra":
    def extractObjectsDataFromXetra(self):
        prefix = self.arg_date + '/'
        objects = list(self.xetraBucket.objects.filter(Prefix=prefix))
        return objects

    ## Método encargado de convertir los objetos en información legible para los DataFrame (y convertirlos a uno):
    def convertObjectsToDataFrame(self, objects):
        data = []

        for obj in objects:
            csv_obj = self.xetraBucket.Object(key=obj.key).get()['Body'].read().decode('utf-8')
            data.append(pd.read_csv(StringIO(csv_obj), delimiter=','))

        df_all = pd.concat(data, ignore_index=True)

        return df_all

class Transform:
    ## Método constructor de la clase:
    def __init__(self, valorEurToMxn):
        self.valorEurToMxn = valorEurToMxn

    ## Método encargado de realizar las modificaciones solicitadas a la información extraida:
    def operationDataFrame(self, df_all):
        # Se seleccionan solo las siguientes columnas:
        columns = ["ISIN", "Mnemonic", "Date", "Time", "StartPrice", "EndPrice", "MinPrice", "MaxPrice", "TradedVolume"]
        df_all = df_all[columns]

        df_all = (df_all
                  .assign(
            opening_price=lambda x: x.sort_values(by='Time').groupby(['ISIN', 'Date'])['StartPrice'].transform('first'),
            closing_price=lambda x: x.sort_values(by='Time').groupby(['ISIN', 'Date'])['EndPrice'].transform('last'))
                  .groupby(['ISIN', 'Date'], as_index=False)
                  .agg(Time=('Time', 'min'),
                       opening_price_eur=('opening_price', 'min'),
                       closing_price_eur=('closing_price', 'min'),
                       minimum_price_eur=('MinPrice', 'min'),
                       maximum_price_eur=('MaxPrice', 'max'),
                       daily_traded_volume=('TradedVolume', 'sum'),
                       StartPrice_eur=('StartPrice', 'min'),
                       EndPrice_eur=('EndPrice', 'min')))

        # Convertir la columna 'Date' y 'Time' en tipo datetime y crear DatetimeIndex:
        # Esto significa que cada fila tendrá como índice la fecha y hora a la que corresponde el registro.
        df_all['datetime'] = pd.to_datetime(df_all['Date'] + ' ' + df_all['Time'])
        df_all = df_all.set_index('datetime')

        # Se especifica el rango de horas a obtener la información, en este caso, de 8:00 AM a 12:00 PM.
        df_all = df_all.between_time('8:0:0', '12:0:0')
        # Se borran las columnas "Date" y "Time" debido a que ya se encuentran visibles en la columna índice "datetime":
        del df_all['Date']
        del df_all['Time']

        # Calcular la desviación estándar entre "StartPrice" y "EndPrice" y agregarla como una nueva columna a df_all:
        df_all['desviacion_estandar'] = df_all[['StartPrice_eur', 'EndPrice_eur']].std(axis=1)

        # Calcular la desviación estándar entre "StartPrice" y "EndPrice" y agregarla como una nueva columna a df_all:
        df_all['EndPrice_mxn'] = df_all['EndPrice_eur'] * self.valorEurToMxn

        # Eliminar missing values:
        # inplace=True elimina los NaN y modifica el df original, si setea como false, creará un nuevo df sin los NaN eliminados.
        df_all.dropna(inplace=True)

        df_all.to_csv('datos.csv')
        print(f"CSV file successfully generated to the root folder {pathlib.Path(__file__).parent.resolve()} with name 'datos.csv'")

        return (df_all)

class Load():
    ## Método constructor de la clase:
    def __init__(self, typeService, target_bucket_name, key):
        self.typeService = typeService
        self.target_bucket_name = target_bucket_name
        self.key = key
    ## Método encargado de recibir el DataFrame transformado y subirlo al bucket especificado:
    def loadObjectToBucket(self, df):
        try:
            out_buffer = BytesIO()
            df.to_parquet(out_buffer, index=False)
            s3 = boto3.client(self.typeService)
            s3.put_object(Body=out_buffer.getvalue(), Bucket=self.target_bucket_name, Key=self.key)
            print(f"Dataframe successfully loaded to {self.target_bucket_name}/{self.key}")

        except Exception as e:
            print(f"Error loading dataframe to {self.target_bucket_name}/{self.key}: {e}")

class Report():

    ## Método constructor de la clase:
    def __init__(self, typeService, target_bucket_name, key):
        self.typeService = typeService
        self.target_bucket_name = target_bucket_name
        self.key = key

    ## Método encargado de extraer la información del Bucket en un DataFrame:
    def getReportOfBucket(self):
        try:
            s3 = boto3.resource(self.typeService)
            bucket_trg = s3.Bucket(self.target_bucket_name)
            prq_obj = bucket_trg.Object(key=self.key).get().get('Body').read()
            data = BytesIO(prq_obj)
            df_report = pd.read_parquet(data)

            print(f"CSV file successfully generated to the root folder {pathlib.Path(__file__).parent.resolve()} with name 'dataReport.csv'")
            return df_report.to_csv('dataReport.csv')

        except Exception as e:
            print(f"Error, it was no possible to get the DataFrame from Bucket -> {e}")
            return None

class Prediction:

    ## Método encargado de realizar la predicción del EndPrice mediante redes neuronáles:
    ## Nota: se utilizó un modelo de red neuronal LSTM.
    def predictionEndPrice(self, dataReportPath):

        # Carga los datos en un DataFrame de pandas
        df = pd.read_csv(dataReportPath)

        # Seleccione solo las columnas necesarias
        data = df.filter(['EndPrice_eur'])

        # Convertir el DataFrame a un numpy array
        dataset = data.values

        # Obtener el número de filas para entrenar el modelo
        training_data_len = math.ceil(len(dataset) * 0.8)
        # Normalizar los datos
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        # Crear el conjunto de datos de entrenamiento
        train_data = scaled_data[0:training_data_len, :]

        # Dividir los datos en x_train y y_train
        x_train = []
        y_train = []

        for i in range(7, len(train_data)):
            x_train.append(train_data[i - 7:i, 0])
            y_train.append(train_data[i, 0])

        # Convertir los datos en numpy arrays
        x_train, y_train = np.array(x_train), np.array(y_train)

        # Dar forma a los datos para que sean aptos para la red neuronal
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Construir el modelo de red neuronal
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))

        # Compilar el modelo
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Entrenar el modelo
        model.fit(x_train, y_train, batch_size=1, epochs=3)

        # Crear el conjunto de datos de prueba
        test_data = scaled_data[training_data_len - 7:, :]

        # Crear el conjunto de datos x_test y y_test
        x_test = []
        y_test = dataset[training_data_len:, :]

        for i in range(7, len(test_data)):
            x_test.append(test_data[i - 7:i, 0])

        # Convertir los datos en numpy arrays
        x_test = np.array(x_test)

        # Dar forma a los datos para que sean aptos para la red neuronal
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        # Realizar una predicción de los precios
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        # Obtener la raíz del error cuadrático medio (RMSE)
        rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
        print('RMSE:', rmse)

        return print(predictions)

## Creación de objeto y llamada de métodos de la clase "Extract" y envío de parámetros:
extract = Extract('s3', 'xetra-1234', '2022-12-31')

## Creación de objeto y llamada de métodos de la clase "ApplicationLayer" y envío de parámetros:
transform = Transform(19.20)

## Creación de llave para indicar el nombre que tendrá el parquet:
key = 'xetra_daily_report_' + datetime.today().strftime("%Y%m%d_%H%M%S") + '.parquet'

## Preparación de DataFrame a subir al Bucket.
df = transform.operationDataFrame(extract.convertObjectsToDataFrame(extract.extractObjectsDataFromXetra()))

## Llamada de método "Load" perteneciente a la capa de aplicación junto a sus respectivos parámetros:
load = Load('s3', 'xetra-aagf', key)
load.loadObjectToBucket(df)
## Llamada de método "Report" para extraer el DataFrame subido recientemente al Bucket determinado:
report = Report('s3', 'xetra-aagf', key)
report.getReportOfBucket()
## Creación de objeto y llamada de métodos de la clase "Prediction" y envío de parámetros:
prediction = Prediction()
prediction.predictionEndPrice('dataReport.csv')


