import pandas as pd
import os
import zipfile as zf


def zip_to_df(path):
    """unzip files and load uncompressed csv files into a list of panda data frames"""

    zip_files = [f for f in os.listdir(path) if f.endswith('.zip')]
    df_objects = list()

    for i in zip_files:
        zip_obj = zf.ZipFile(file=path + i)
        filename = zip_obj.namelist()[0]
        print('unzipping {}...'.format(filename))
        df = pd.read_csv(filepath_or_buffer=zip_obj.open(filename), index_col='id')
        zip_obj.close()
        df_objects.append(df)

    return df_objects

