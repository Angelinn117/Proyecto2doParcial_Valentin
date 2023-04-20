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

## Capa Adaptativa:
class AdapterLayer:

    ## Método constructor de la clase:
    def __init__(self, typeService, xetraBucket, arg_date):
        self.typeService = boto3.resource(typeService)
        self.xetraBucket = self.typeService.Bucket(xetraBucket)
        self.arg_date = arg_date

    ## Método encargado de extraer información de la base de datos xetra-1234:
    def extractObjectsDataFromXetra(self):
        # Se convierte la fecha Date en tipo "DateTime" donde se especifica la fecha y la hora:
        arg_date_dt = datetime.strptime(self.arg_date, '%Y-%m-%d').date() - timedelta(days=1)

        # Se almacenan los objetos obtenidos del bucket mediante un ciclo y una fecha específica:
        objects = [obj for obj in self.xetraBucket.objects.all() if
                   datetime.strptime(obj.key.split("/")[0], '%Y-%m-%d').date() >= arg_date_dt]

        return objects

    ## Método encargado de convertir los objetos obtenidos en un DataFrame:
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
            print (df_all)

        return df_all




## Llamada de métodos de la clase "AdapterLayer":
adapterLayer = AdapterLayer('s3', 'xetra-1234', '2022-12-31')

# Llamada de métodos mediante objeto "adaptarLayer", así como el respectivo envío de parámetros:
adapterLayer.convertObjectsToDataFrame(adapterLayer.extractObjectsDataFromXetra())