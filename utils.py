import pandas as pd
import os
import zipfile as zf


def unzip_folder(path):
    """read zip files and load uncompressed csv files into a list of panda dataframes"""

    zip_files = [f for f in os.listdir(path) if f.endswith('.zip')]
    list_dataframes = list()

    for i in zip_files:
        zip_obj = zf.ZipFile(file=path + i)
        filename = zip_obj.namelist()[0]
        print('unzipping {}...'.format(filename))
        df = pd.read_csv(filepath_or_buffer=zip_obj.open(filename), index_col='id')
        zip_obj.close()
        list_dataframes.append(df)

    return list_dataframes

