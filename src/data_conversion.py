"""

"""
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from lists import columns_to_load

path_to_csv_file = './../data/train.csv'

# read the CSV file using dask
ddf = dd.read_csv(path_to_csv_file, usecols=columns_to_load)

# repartition to a single partition
ddf = ddf.repartition(npartitions=1)

ddf.head()

# path for the output Parquet file
path_to_parquet_output = './../data/single_file.parquet'
# setup the progress bar
pbar = ProgressBar()
pbar.register()

# convert the dask df to a single parquet file
ddf.to_parquet(
    path_to_parquet_output,
    engine='pyarrow',
    compression='snappy'
)
