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

extract = Extract('s3', 'xetra-1234', '2022-12-31')
df = extract.convertObjectsToDataFrame(extract.extractObjectsDataFromXetra())
print(df)
