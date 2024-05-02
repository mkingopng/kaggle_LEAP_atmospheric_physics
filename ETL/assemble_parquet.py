"""
This script reads all Parquet files in a directory and concatenates them into
a single Parquet file
"""
import pandas as pd
import pyarrow.parquet as pq
import os
from tqdm import tqdm


def assemble_parquet_files(directory, output_file):
    """
    Read all Parquet files in a directory and concatenate them into a single
    Parquet file
    :param directory:
    :param output_file:
    :return:
    """
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.parquet')]
    dfs = []
    for file in tqdm(files, desc="Reading Parquet files"):
        df = pq.read_table(file).to_pandas()
        dfs.append(df)
    print("Concatenating DataFrames...")
    full_df = pd.concat(dfs, ignore_index=True)
    print("Writing to output file...")
    full_df.to_parquet(output_file)
    print("Done!")


if __name__ == '__main__':
    assemble_parquet_files(
        './../data/parquet_files',
        'train_df.parquet'
    )
