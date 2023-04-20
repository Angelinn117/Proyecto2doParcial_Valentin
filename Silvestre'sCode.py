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


class Layer:
    def __init__(self, typeService, xetraBucket, arg_date):
        self.typeService = boto3.resource(typeService)
        self.xetraBucket = self.typeService.Bucket(xetraBucket)
        self.arg_date = arg_date

    # Método encargado de extraer información de la base de datos xetra-1234:
    def extractObjectsDataFromXetra(self):
        # Se convierte la fecha Date en tipo "DateTime" donde se especifica la fecha y la hora:
        arg_date_dt = datetime.strptime(self.arg_date, '%Y-%m-%d').date() - timedelta(days=1)

        # Se almacenan los objetos obtenidos del bucket mediante un ciclo y una fecha específica:
        objects = [obj for obj in self.xetraBucket.objects.all() if
                   datetime.strptime(obj.key.split("/")[0], '%Y-%m-%d').date() >= arg_date_dt]

        return objects

    # Método encargado de convertir los objetos obtenidos en un DataFrame:
    def convertObjectsToDataFrame(self, objects):
        # Se obtiene el objeto inicial por posición en el bucket objects:
        csv_obj_init = self.xetraBucket.Object(key=objects[0].key).get().get('Body').read().decode('utf-8')
        data = StringIO(csv_obj_init)
        df_init = pd.read_csv(data, delimiter=',')

        # Concatenar todos los objetos con pandas al DataFrame:
        df_all = pd.DataFrame(columns=df_init.columns)
        for obj in objects:
            csv_obj = self.xetraBucket.Object(key=obj.key).get().get('Body').read().decode('utf-8')
            data = StringIO(csv_obj)
            df = pd.read_csv(data, delimiter=',')
            df_all = pd.concat([df, df_all], ignore_index=True)

        return df_all

    # Método encargado de transofmar la informacion:
    def transformation(self, df_all):
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
        # ct = ColumnTransformer([
        #     ('poly', poly, [1])  # apply poly_transformer to column 1 (the second column)
        # ])
        #
        # ct.fit(X)
        # X_Poly = ct.transform(X)


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

    # Método encargado de exportar los dataframe al bucket:
    def load(self, df):
        out_buffer = BytesIO()
        df.to_parquet(out_buffer, index=False)
        s3 = boto3.client(self.typeServices)
        s3.put_object(Body=out_buffer.getvalue(), Bucket=self.target_bucket_name, Key=self.key)

    # Método encargado de traer los datos del bucket y mostrarlos:
    def report(self):

        s3 = boto3.resource(self.typeServices)
        bucket_trg = s3.Bucket(self.target_bucket_name)
        prq_obj = bucket_trg.Object(key=self.key).get().get('Body').read()
        data = BytesIO(prq_obj)
        df_report = pd.read_parquet(data)
        for obj in bucket_trg.objects.all():
            print(obj.key)

        print("DataFrame obtenido de Bucket:")
        print(df_report)


class AdaptiveLayer(Layer):
    def __init__(self, typeService, xetraBucket, arg_date):
        super().__init__(typeService, xetraBucket, arg_date)

    def extractObjectsDataFromXetra(self):
        return super().extractObjectsDataFromXetra()

    def convertObjectsToDataFrame(self, objects):
        return super().convertObjectsToDataFrame(objects)


class ApplicationLayer(Layer):
    def __init__(self, valorEurToMxn, typeServices, key, target_bucket_name):
        self.valorEurToMxn = valorEurToMxn
        self.typeServices = typeServices
        self.key = key
        self.target_bucket_name = target_bucket_name

    def transformation(self, df_all):
        return super().transformation(df_all)

    def load(self, df):
        super().load(df)

    def report(self):
        super().report()

# instancias de clases y metodos con una estructura ETL
adap_layer = AdaptiveLayer('s3', 'xetra-1234',  '2022-12-31')
objects = adap_layer.extractObjectsDataFromXetra()
df_all = adap_layer.convertObjectsToDataFrame(objects)

key = 'xetra_daily_report_' + datetime.today().strftime("%Y%m%d_%H%M%S") + '.parquet'
app_layer = ApplicationLayer(19.20,'s3',key,'xetra-sjcb')
df_transf = app_layer.transformation(df_all)
app_layer.load(df_transf)

print("prueba de cambio.")