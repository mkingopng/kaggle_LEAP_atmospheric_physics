"""

"""
import boto3
from botocore.exceptions import ClientError

iam = boto3.client('iam')
glue = boto3.client('glue')

# Set up the logging parameters
log_group_name = "Michael_Kingston_Dev_Gru"
log_stream_prefix = "GlueJobs"


def create_iam_role(role_name, policy_arn):
    """
    create an IAM role for AWS Glue
    :param role_name:
    :param policy_arn:
    :return:
    """
    try:
        role = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument='''{
                "Version": "2012-10-17",
                "Statement": [
                    {
                    "Effect": "Allow", 
                    "Principal": {"Service": "glue.amazonaws.com"}, 
                    "Action": "sts:AssumeRole"
                    }
                ]
            }''',
            Description="Role for AWS Glue to access resources"
        )
        iam.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
        return role['Role']['Arn']
    except ClientError as error:
        print(f"Failed to create role: {error}")
        return None


def create_glue_crawler(crawler_name, role_arn, database_name, s3_path):
    """

    :param crawler_name:
    :param role_arn:
    :param database_name:
    :param s3_path:
    :return:
    """
    glue = boto3.client('glue')
    glue.create_crawler(
        Name=crawler_name,
        Role=role_arn,
        DatabaseName=database_name,
        Targets={'S3Targets': [{'Path': s3_path}]}
    )
    glue.start_crawler(Name=crawler_name)


def setup_glue_job(name, role, script_location, temp_dir, additional_args, input_path, output_path):
    """
    Configure and create the Glue job
    :param name:
    :param role:
    :param script_location:
    :param temp_dir:
    :param additional_args:
    :param input_path:
    :param output_path:
    :return:
    """
    # Add input and output paths to the job arguments
    additional_args['--INPUT_PATH'] = input_path
    additional_args['--OUTPUT_PATH'] = output_path

    # Add logging settings
    additional_args['--enable-continuous-cloudwatch-log'] = 'true'
    additional_args['--enable-continuous-log-filter'] = 'true'
    additional_args['--continuous-log-logGroup'] = log_group_name
    additional_args['--continuous-log-logStreamPrefix'] = log_stream_prefix

    try:
        return glue.create_job(
            Name=name,
            Role=role,
            Command={
                'Name': 'glueetl',
                'ScriptLocation': script_location,
                'PythonVersion': '3'
            },
            DefaultArguments={'--TempDir': temp_dir, **additional_args},
            GlueVersion='3.0',
            WorkerType='Standard',
            NumberOfWorkers=10,
            ExecutionProperty={'MaxConcurrentRuns': 1}
        )
    except ClientError as error:
        print(f"Failed to create job: {error}")
        return None


def start_job(name):
    """
    start the specified Glue job
    :param name:
    :return:
    """
    try:
        job_run = glue.start_job_run(JobName=name)
        print(f"Job started with run ID: {job_run['JobRunId']}")
        return job_run['JobRunId']
    except ClientError as error:
        print(f"Failed to start job: {error}")
        return None


def get_job_status(name, run_id):
    """
    retrieve the status of a Glue job run
    :param name:
    :param run_id:
    :return:
    """
    try:
        status = glue.get_job_run(JobName=name, RunId=run_id)
        return status['JobRun']['JobRunState']
    except ClientError as error:
        print(f"Failed to get job status: {error}")
        return None


if __name__ == "__main__":
    # configuration variables (set via environment variables or config)
    job_name = "CSVToParquetConversion"
    role_arn = "arn:aws:iam::001499655372:role/GlueSuperUser"
    script_location = "s3://csvtoparquet1/transform_script_v1.py"
    temp_dir = "s3://csvtoparquet1/temp/glue/"

    # create a job
    job_details = setup_glue_job(
        job_name,
        role_arn,
        script_location,
        temp_dir,
        {
            '--job-bookmark-option': 'job-bookmark-enable',
            '--enable-metrics': '',
            '--additional-python-modules': 'pyarrow==4.0.0,awswrangler==2.8.0'
        },
        input_path="s3://csvtoparquet1/",
        output_path="s3://csvtoparquet1/output/"
    )

    # start the job
    run_id = start_job(job_name)

    # check job status
    if run_id:
        status = get_job_status(job_name, run_id)
        print(f"Job run status: {status}")
