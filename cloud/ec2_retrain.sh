#!/usr/bin/env bash
set -euo pipefail

# EC2 user-data/cron entry point. Configure the repository and AWS CLI before
# using this script. The existing training pipeline is intentionally reused.
cd /opt/pdm

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "candidate_runs/${RUN_ID}"

# Download the current production artifacts without deleting the repository.
aws s3 sync "s3://${PDM_S3_BUCKET}/production/models/" models/ || true
aws s3 sync "s3://${PDM_S3_BUCKET}/production/scalers/" scalers/ || true

# Retraining writes into the candidate workspace. It is uploaded separately;
# production is promoted only after the RMSE comparison is reviewed.
python main.py
cp -r models "candidate_runs/${RUN_ID}/models"
cp -r outputs "candidate_runs/${RUN_ID}/outputs"
cp -r logs "candidate_runs/${RUN_ID}/logs"

# Upload a versioned candidate, never directly over production.
aws s3 sync "candidate_runs/${RUN_ID}/" "s3://${PDM_S3_BUCKET}/candidates/${RUN_ID}/"
echo "Candidate uploaded: ${RUN_ID}"
