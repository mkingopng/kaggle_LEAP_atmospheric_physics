"""
Glue job to assemble parquets
"""
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

args = getResolvedOptions(sys.argv, ['JOB_NAME', 'INPUT_PATH', 'OUTPUT_PATH'])

def main():
    # Initialize a Glue context
    sc = SparkContext()
    glueContext = GlueContext(sc)
    spark = glueContext.spark_session
    job = Job(glueContext)
    job.init(args['JOB_NAME'], args)

    # Read Parquet files from the specified S3 path
    datasource0 = glueContext.create_dynamic_frame.from_options(
        connection_type = "s3",
        format = "parquet",
        connection_options = {"paths": [args['INPUT_PATH']]},
        transformation_ctx = "datasource0"
    )

    # Convert to DataFrame for operations
    df = datasource0.toDF()

    # If you need to perform operations, you can do them here using DataFrame APIs
	# todo: drop unneeded columns

    # Convert back to DynamicFrame if necessary
    dynamic_frame_write = DynamicFrame.fromDF(df, glueContext, "dynamic_frame_write")

    # Write out the data in Parquet format to the output path
    datasink = glueContext.write_dynamic_frame.from_options(
        frame = dynamic_frame_write,
        connection_type = "s3",
        connection_options = {"path": args['OUTPUT_PATH']},
        format = "parquet",
        transformation_ctx = "datasink"
    )

    job.commit()

if __name__ == "__main__":
    main()

