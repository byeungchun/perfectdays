$configs = @(
    "C:\Users\by003457\workspace\perfectdays\config\strategy01_aggressive.yaml",
    "C:\Users\by003457\workspace\perfectdays\config\strategy01_conservative.yaml",
    "C:\Users\by003457\workspace\perfectdays\config\strategy01_creative.yaml"
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$runnerScript = Join-Path $scriptDir 'run-strategy01.ps1'

if (-not (Test-Path $runnerScript)) {
    throw "Runner script not found at $runnerScript"
}

$jobs = @()
foreach ($cfg in $configs) {
    $jobName = "strategy01_" + [System.IO.Path]::GetFileNameWithoutExtension($cfg)
    Write-Host "Queueing config: $cfg" -ForegroundColor Cyan
    $jobs += Start-Job -Name $jobName -ArgumentList $runnerScript, $cfg -ScriptBlock {
        param($runnerScriptInner, $configPath)
        try {
            Write-Host "Starting config: $configPath" -ForegroundColor Cyan
            & $runnerScriptInner -ConfigPath $configPath
        }
        catch {
            Write-Error "Run failed for $configPath: $_"
            throw
        }
    }
}

if ($jobs.Count -eq 0) {
    Write-Warning "No jobs started. Check config list."
    return
}

Write-Host "Waiting for all strategy runs to complete..." -ForegroundColor Yellow
Wait-Job -Job $jobs

$failed = @()
foreach ($job in $jobs) {
    Write-Host "Results for $($job.Name):" -ForegroundColor Green
    Receive-Job -Job $job
    if ($job.State -ne 'Completed') {
        $failed += $job
        Write-Warning "Job $($job.Name) ended in state $($job.State)."
    }
}

Remove-Job -Job $jobs -Force

if ($failed.Count -gt 0) {
    throw "One or more strategy runs failed. See warnings above."
}

Write-Host "All strategy runs completed successfully." -ForegroundColor Green