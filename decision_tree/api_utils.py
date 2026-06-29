import configparser
import os
from urllib.parse import urlparse
from gri_shared_library.os_tools import is_file_recent, create_folder, get_project_root_dir
from gri_shared_library.s3_tools import get_aws_session
from tm_api_utils import pull_tm_api_data


def download_geoparquet(params, secrets, tm_raw):
    """
    Download a geoparquet file from S3 with boto3, load it with GeoPandas,
    and save it as a CSV.

    Assumptions:
    - params["outfile"]["tm_geoparquet_source"] stores the S3 path
      like: s3://bucket/path/to/file.geoparquet
    - tm_outfile is the CSV output path
    - AWS auth through console credentials and profile
    """
    s3_uri = params["s3"].get("geoparquet")
    if "data_version" in s3_uri:
        data_v = params["outfile"].get("data_version")
        month, day, year = data_v.split("-")
        data_v_reformat = f"{year}-{month}-{day}"
        s3_url = params["s3"].get("geoparquet").format(data_version=data_v_reformat)
    else:
        s3_url = params["s3"].get("geoparquet")

    aws_profile = secrets.get("aws", {}).get("aws_profile")
    print(s3_url)

    # Parse s3:// url to get bucket/key
    parsed = urlparse(s3_url)
    if parsed.scheme != "s3":
        raise ValueError(f"Expected s3:// URL, got: {s3_url}")

    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    # Create target folder
    target_folder = os.path.dirname(tm_raw)
    create_folder(target_folder)

    # Retrieve parquet file from S3
    aws_session = get_aws_session(profile_name=aws_profile)
    s3_client = aws_session.client("s3")
    s3_client.download_file(bucket, key, tm_raw)
    print(f"Downloaded to {tm_raw}")

def aws_profile_exists(profile_name: str) -> bool:
    """
    Check if an AWS CLI profile exists in ~/.aws/config or ~/.aws/credentials.

    Args:
        profile_name (str): The AWS profile name to check.

    Returns:
        bool: True if the profile exists, False otherwise.
    """
    if not profile_name or not isinstance(profile_name, str):
        raise ValueError("Profile name must be a non-empty string.")

    # AWS config and credentials file paths
    aws_dir = os.path.expanduser("~/.aws")
    config_path = os.path.join(aws_dir, "config")
    credentials_path = os.path.join(aws_dir, "credentials")

    parser = configparser.ConfigParser()

    # Check credentials file
    if os.path.exists(credentials_path):
        parser.read(credentials_path)
        if profile_name in parser.sections():
            return True

    # Check config file (profiles are prefixed with 'profile ' except 'default')
    if os.path.exists(config_path):
        parser.read(config_path)
        if profile_name == "default" and "default" in parser.sections():
            return True
        if f"profile {profile_name}" in parser.sections():
            return True

    return False
