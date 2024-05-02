import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job


def main():
	# initialize a Glue context
	glueContext = GlueContext(SparkContext.getOrCreate())

	# read arguments passed when starting the job
	args = getResolvedOptions(sys.argv, ['JOB_NAME', 'INPUT_PATH', 'OUTPUT_PATH'])

	# create a DynamicFrame using the Glue context and the input path
	datasource0 = glueContext.create_dynamic_frame.from_options(
		connection_type="s3",
		connection_options={"paths": [args['INPUT_PATH']]},
		format="csv",
		format_options={"withHeader": True}
	)

	# write out the data in Parquet format to the output path
	datasink4 = glueContext.write_dynamic_frame.from_options(
		frame=datasource0,
		connection_type="s3",
		connection_options={"path": args['OUTPUT_PATH']},
		format="parquet"
	)


# entry point for the script
if __name__ == '__main__':
	main()