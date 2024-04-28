"""

"""
import boto3
import os
from tqdm import tqdm
from botocore.exceptions import NoCredentialsError


def upload_file_to_s3(file_name, bucket, object_name=None):
    """
    Upload a file to an S3 bucket with progress bar

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified, use a sanitized file name
    :return: True if file was uploaded, else False
    """
    # Sanitize file_name to use as object_name if none provided
    if object_name is None:
        object_name = file_name.split('/')[-1]  # This assumes a POSIX path format

    # Create an S3 client
    s3_client = boto3.client('s3')

    # Get the size of the file
    file_size = os.path.getsize(file_name)

    # Create a tqdm progress bar
    with tqdm(total=file_size, unit='B', unit_scale=True, desc=file_name) as pbar:
        def upload_progress(chunk):
            pbar.update(chunk)

        try:
            # Upload the file with a progress bar
            s3_client.upload_file(
                Filename=file_name,
                Bucket=bucket,
                Key=object_name,
                Callback=upload_progress
            )
        except NoCredentialsError:
            print("Credentials not available")
            return False
        except Exception as e:
            print(f"An error occurred: {e}")
            return False

    return True

# Example usage
file_name = './../data/train.csv'
bucket_name = 'csvtoparquet1'

success = upload_file_to_s3(file_name, bucket_name)

if success:
    print("File uploaded successfully.")
else:
    print("File upload failed.")
