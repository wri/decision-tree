import yaml
import numpy as np
import boto3
import os

def upload_to_s3(params, config_path, project_name, today):
    '''
    s3://restoration-monitoring/tree_verification/output/project_data/project_shortname/geojson
    '''
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    aak = config['aws']['aws_access_key_id']
    ask = config['aws']['aws_secret_access_key']
    bucket = params["s3"]["bucket_name"]
    folder_prefix = params["s3"]["folder_prefix"]
    filename = f"{project_name}_{today}.geojson" 
    folder_path = f"{folder_prefix}/{project_name}/geojson/"
    s3_key = f"{folder_path}{filename}"
    local_path = os.path.join(params["outfile"]["geojsons"], filename)
    
    s3 = boto3.client('s3', 
                      aws_access_key_id=aak, 
                      aws_secret_access_key=ask)

    # Check if the project folder exists
    result = s3.list_objects_v2(Bucket=bucket, Prefix=folder_path, MaxKeys=1)
    if 'Contents' not in result:
        s3.put_object(Bucket=bucket, Key=folder_path)

    s3.upload_file(local_path, bucket, s3_key)