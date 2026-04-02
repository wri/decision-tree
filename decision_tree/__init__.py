# initialize aws
import os
from pathlib import Path

import boto3

if (
        "AWS_ACCESS_KEY_ID" in os.environ
        and "AWS_SECRET_ACCESS_KEY" in os.environ
        and "AWS_SESSION_TOKEN" in os.environ
):
    aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
    aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    aws_session_token = os.environ["AWS_SESSION_TOKEN"]
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
        region_name='us-east-1'
    )
    S3_CLIENT = session.client('s3')

else:
    if "AWS_PROFILE" in os.environ:
        aws_profile = os.environ["AWS_PROFILE"]
    else:
        aws_profile = "LandResearchUser"

    credentials_file_path = Path(os.path.join(Path.home(), '.aws', 'credentials'))
    config_file_path = Path(os.path.join(Path.home(), '.aws', 'config'))

    if credentials_file_path.exists() or config_file_path.exists():
        try:
            session = boto3.Session(profile_name=aws_profile, region_name='us-east-1')
            S3_CLIENT = session.client('s3')
        except Exception as e:
            raise Exception(f"Could not initialize S3 client with profile '{aws_profile}': {e}")
    else:
        try:
            session = boto3.Session()
            S3_CLIENT = session.client('s3')
        except Exception as e:
            raise Exception(f"Could not initialize S3 client without a profile: {e}")