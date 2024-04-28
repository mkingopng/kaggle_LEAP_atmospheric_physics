"""

"""
import boto3
from botocore.exceptions import ClientError

iam = boto3.client('iam')  # fix_me
glue = boto3.client('glue')  # fix_me


def create_iam_role(role_name, policy_arn):
    """Create an IAM role for AWS Glue."""
    try:
        role = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument='''{
                "Version": "2012-10-17",
                "Statement": [
                    {"Effect": "Allow", "Principal": {"Service": "glue.amazonaws.com"}, "Action": "sts:AssumeRole"}
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


def setup_glue_job(name, role, script_location, temp_dir, additional_args):
    """Configure and create the Glue job."""
    try:
        return glue.create_job(
            Name=name,
            Role=role,
            Command={'Name': 'glueetl', 'ScriptLocation': script_location, 'PythonVersion': '3'},
            DefaultArguments={'--TempDir': temp_dir, **additional_args},
            GlueVersion='2.0',
            WorkerType='Standard',
            NumberOfWorkers=10,
            ExecutionProperty={'MaxConcurrentRuns': 1}
        )
    except ClientError as error:
        print(f"Failed to create job: {error}")
        return None


def start_job(name):
    """Start the specified Glue job."""
    try:
        job_run = glue.start_job_run(JobName=name)
        print(f"Job started with run ID: {job_run['JobRunId']}")
        return job_run['JobRunId']
    except ClientError as error:
        print(f"Failed to start job: {error}")
        return None


def get_job_status(name, run_id):
    """Retrieve the status of a Glue job run."""
    try:
        status = glue.get_job_run(JobName=name, RunId=run_id)
        return status['JobRun']['JobRunState']
    except ClientError as error:
        print(f"Failed to get job status: {error}")
        return None


if __name__ == "__main":
    # configuration variables (set via environment variables or config)
    job_name = "CSVToParquetConversion"  # fix_me
    role_arn = "your-iam-role-arn"  # fix_me
    script_location = "s3://your-script-bucket/scripts/transform_script.py"  # fix_me
    temp_dir = "s3://your-temporary-bucket/temp/"  # fix_me

    # create a job
    job_details = setup_glue_job(
        job_name,
        role_arn,
        script_location,
        temp_dir,
        {
            '--job-bookmark-option': 'job-bookmark-enable',
            '--enable-metrics': '',
            '--additional-python-modules': 'pyarrow==2,awswrangler==2.8.0'
        })

    # start the job
    run_id = start_job(job_name)

    # check job status
    if run_id:
        status = get_job_status(job_name, run_id)
        print(f"Job run status: {status}")
