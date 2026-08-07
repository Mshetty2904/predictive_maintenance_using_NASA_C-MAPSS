param(
    [ValidateSet("normal", "small", "medium", "severe", "all")]
    [string]$Severity = "all",
    [int]$MaxEngines = 0,
    [double]$NoiseStd = 0.02
)

$datasets = @("FD001", "FD002", "FD003", "FD004")
foreach ($dataset in $datasets) {
    $arguments = @("drift_monitor.py", "--dataset", $dataset, "--severity", $Severity, "--noise-std", "$NoiseStd")
    if ($MaxEngines -gt 0) {
        $arguments += @("--max-engines", "$MaxEngines")
    }
    Write-Host "Running calibrated drift monitoring for $dataset..."
    python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Drift monitoring failed for $dataset"
    }
}
