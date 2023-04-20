import boto3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from io import StringIO, BytesIO
from datetime import datetime, timedelta

class Extract:
    def __init__(self, typeService, xetraBucket, arg_date):
        self.typeService = boto3.resource(typeService)
        self.xetraBucket = self.typeService.Bucket(xetraBucket)
        self.arg_date = arg_date

    def extractObjectsDataFromXetra(self):
        prefix = self.arg_date + '/'
        objects = list(self.xetraBucket.objects.filter(Prefix=prefix))
        return objects

    def convertObjectsToDataFrame(self, objects):
        data = []
        for obj in objects:
            csv_obj = self.xetraBucket.Object(key=obj.key).get()['Body'].read().decode('utf-8')
            data.append(pd.read_csv(StringIO(csv_obj), delimiter=','))

        df_all = pd.concat(data, ignore_index=True)

        return df_all

## Capa de Aplicación:
class Transform:

    ## Método constructor de la clase:
    def __init__(self, valorEurToMxn):
        self.valorEurToMxn = valorEurToMxn

    def operationDataFrame(self, df_all):
        # Se seleccionan solo las siguientes columnas:
        columns = ["ISIN", "Mnemonic", "Date", "Time", "StartPrice", "EndPrice", "MinPrice", "MaxPrice", "TradedVolume"]
        df_all = df_all.loc[:, columns]

        # Se creará una nueva columna en el df con el precio de apertura.
        df_all['opening_price'] = df_all.sort_values(by=['Time']).groupby(['ISIN', 'Date'])['StartPrice'].transform(
            'first')

        # Se creará una nueva columna en el df con el precio de cierre.
        df_all['closing_price'] = df_all.sort_values(by=['Time']).groupby(['ISIN', 'Date'])['EndPrice'].transform(
            'last')

        # Se agrupa la informacion del df_all por "ISIN y Date":
        # Nota, se agregó la columna de "Time", "StartPrice" y "EndPrice".
        df_all = df_all.groupby(['ISIN', 'Date'],
                                as_index=False).agg(Time=('Time', 'min'),
                                                    opening_price_eur=('opening_price', 'min'),
                                                    closing_price_eur=('closing_price', 'min'),
                                                    minimum_price_eur=('MinPrice', 'min'),
                                                    maximum_price_eur=('MaxPrice', 'max'),
                                                    daily_traded_volume=('TradedVolume', 'sum'),
                                                    StartPrice_eur=('StartPrice', 'min'),
                                                    EndPrice_eur=('EndPrice', 'min'))

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

        print(df_all)

        # Se crea una ventana de tiempo
        start_time = '2022-12-31 08:00:00'
        end_time = '2022-12-31 12:00:00'
        df_window = df_all.loc[start_time:end_time]

        # Seleccionar las características relevantes y la variable objetivo.
        X = df_window[
            ['opening_price_eur', 'minimum_price_eur', 'maximum_price_eur', 'daily_traded_volume', 'StartPrice_eur']]
        y = df_window['EndPrice_eur']

        # Dividir los datos en conjuntos de entrenamiento 30% prueba y 70% entrenamiento.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Transformar las características seleccionadas a una forma polinómica
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        # Crear un objeto de regresión lineal y ajustar el modelo con los datos de entrenamiento
        model = LinearRegression()
        model.fit(X_train_poly, y_train)

        # Evaluar el rendimiento del modelo utilizando el conjunto de prueba
        score = model.score(X_test_poly, y_test)
        print(f'score: {score}')

        # Utilizar el modelo para hacer predicciones para datos nuevos
        # 'opening_price_eur', 'minimum_price_eur', 'maximum_price_eur', 'daily_traded_volume', 'StartPrice_eur'
        new_data = np.array([[50.0, 40.0, 60.0, 10000.0, 45.0]])
        new_data_poly = poly.transform(new_data)
        predicted_price = model.predict(new_data_poly)

        print(f'Predicted EndPrice_eur: {predicted_price[0]}')

        # opening_price_eur
        plt.scatter(X_train_poly[:, 1], y_train)
        plt.xlabel('x1')
        plt.ylabel('EndPrice_eur')
        plt.show()

        return (df_all)

## Llamada de métodos de la clase "Extract":
extract = Extract('s3', 'xetra-1234', '2022-12-31')

## Llamada de métodos de la clase "ApplicationLayer":
transform = Transform(19.20)

## Creación de llave para indicar el nombre que tendrá el parquet así como enviar el DataFrame a subir al Bucket.
key = 'xetra_daily_report_' + datetime.today().strftime("%Y%m%d_%H%M%S") + '.parquet'
df = transform.operationDataFrame(extract.convertObjectsToDataFrame(extract.extractObjectsDataFromXetra()))
