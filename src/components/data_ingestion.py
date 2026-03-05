import os
import pandas as pd
from sklearn.model_selection import train_test_split

class DataIngestion:
    def Ingest_data(self, path):
        data = pd.read_csv(path)
        return data
