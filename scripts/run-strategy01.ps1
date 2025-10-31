[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$EnvPath,
    [string]$OutputDir = (Join-Path $env:USERPROFILE "Downloads"),
    [int]$MinimumRecords = 20,
    [int]$MinAgentDays = 240,
    [double]$Budget = 100000000,
    [double]$ZVol = 1.0,
    [double]$VwapRelStdMax = 0.005,
    [string[]]$Tickers = @(),
    [switch]$AllTickers,
    [switch]$EnhancedSell,
    [int]$MaxWorkers = 0
)

# Example usage:
# .\run-strategy01.ps1 `
#   -EnvPath C:\Users\byeun\workspace\perfectdays\.env `
#   -OutputDir C:\Users\byeun\Downloads `
#   -Tickers A010420,A005930 `
#   -EnhancedSell `
#   -MaxWorkers 4

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir

if (-not $EnvPath) {
    $EnvPath = Join-Path $repoRoot ".env"
}

$arguments = @(
    "-m", "backtests.strategy01.eda_strategy01_agent",
    "--env-path", $EnvPath,
    "--output-dir", $OutputDir,
    "--minimum-records", $MinimumRecords,
    "--min-agent-days", $MinAgentDays,
    "--budget", $Budget,
    "--z-vol", $ZVol,
    "--vwap-rel-std-max", $VwapRelStdMax
)

if ($Tickers.Count -gt 0) {
    foreach ($ticker in $Tickers) {
        $arguments += @("--ticker", $ticker)
    }
}

if ($AllTickers.IsPresent) {
    $arguments += "--all-tickers"
}

if ($EnhancedSell.IsPresent) {
    $arguments += "--enhanced-sell"
}

if ($MaxWorkers -gt 0) {
    $arguments += @("--max-workers", $MaxWorkers)
}

Push-Location $repoRoot
try {
    Write-Verbose "Running: $Python $($arguments -join ' ')"
    & $Python @arguments
}
finally {
    Pop-Location
}
