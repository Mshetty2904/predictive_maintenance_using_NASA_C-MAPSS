"""AWS Lambda trigger for an EC2 retraining job.

Environment variables:
  EC2_INSTANCE_ID: instance to start
  AWS_REGION: optional AWS region

The Lambda function only starts EC2. Training and model promotion remain on
EC2, keeping cloud orchestration separate from the existing main.py pipeline.
"""

import json
import os

import boto3


def lambda_handler(event, context):
    instance_id = os.environ["EC2_INSTANCE_ID"]
    ec2 = boto3.client("ec2", region_name=os.getenv("AWS_REGION"))
    response = ec2.start_instances(InstanceIds=[instance_id])
    return {
        "statusCode": 202,
        "body": json.dumps({"instance_id": instance_id, "state": response["StartingInstances"]}),
    }
