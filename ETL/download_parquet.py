"""
Download an entire directory of files parquet from S3 to a local directory
"""
import boto3
import os
from tqdm import tqdm

# Create an S3 client
s3 = boto3.client('s3')


def download_directory_from_s3(bucket, remote_directory_name, local_directory_name):
    """
    Download an entire directory of files from S3 to a local directory
    :param bucket:
    :param remote_directory_name:
    :param local_directory_name:
    :return:
    """
    paginator = s3.get_paginator('list_objects')
    for result in paginator.paginate(Bucket=bucket, Prefix=remote_directory_name):
        contents = result.get('Contents', [])
        for content in tqdm(contents, desc='Downloading files'):
            # Calculate relative path
            rel_path = content['Key'][len(remote_directory_name):]
            local_path = os.path.join(local_directory_name, rel_path)

            # ensure local directory exists
            local_dir = os.path.dirname(local_path)
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)

            # download the file
            try:
                s3.download_file(bucket, content['Key'], local_path)
            except Exception as e:
                print(f"Failed to download {content['Key']}: {e}")


if __name__ == '__main__':
    download_directory_from_s3(
        'csvtoparquet1',
        'output/',
        './../data/parquet_files'
    )
