# AWS deployment skeleton

This project uses three AWS services:

1. S3 stores model versions, metrics, SHAP outputs, drift reports, and logs.
2. Lambda receives a drift trigger and starts the configured EC2 instance.
3. EC2 runs `main.py`, evaluates the candidate model, and uploads artifacts.

## Deployment sequence

1. Create an S3 bucket and prefixes `production/`, `candidates/`, and
   `reports/`.
2. Launch an EC2 instance with Python and the project dependencies installed.
3. Attach an EC2 IAM role with S3 read/write access and copy the repository to
   `/opt/pdm`.
4. Set `PDM_S3_BUCKET` and run `cloud/ec2_retrain.sh`.
5. The script uploads a timestamped candidate under
   `s3://BUCKET/candidates/<run-id>/`; it never overwrites production.
6. Compare candidate validation RMSE against the current production model.
   Promote only a lower-RMSE candidate by copying its model/scaler artifacts
   to the `production/` prefix and updating a `current.json` pointer.
7. Configure Lambda with `EC2_INSTANCE_ID` and invoke it only after a drift
   decision has `action: trigger_retraining`.

Example local monitoring commands:

```powershell
python drift_monitor.py --dataset FD001 --severity all --noise-std 0.02
.\run_drift_all.ps1 -Severity all -NoiseStd 0.02
```

Use `-MaxEngines 5` for a quick smoke test. For dissertation results, run all
engines and report normal false-positive rate, detection rate by severity, and
the number of consecutive batches required before retraining is triggered.

Required environment variables:

- Lambda: `EC2_INSTANCE_ID`, optional `AWS_REGION`
- EC2: `PDM_S3_BUCKET`

The Lambda role needs `ec2:StartInstances`. The EC2 instance role needs S3
read/write permissions for the project bucket. Do not commit AWS credentials;
use IAM roles and environment configuration.
