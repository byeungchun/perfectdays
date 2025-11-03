[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$EnvPath,
    [string]$OutputDir,
    [int]$MinimumRecords = 20,
    [int]$MinAgentDays = 240,
    [Nullable[Double]]$Budget,
    [Nullable[Double]]$ZVol,
    [Nullable[Double]]$VwapRelStdMax,
    [string[]]$Tickers = @(),
    [switch]$AllTickers,
    [switch]$EnhancedSell,
    [switch]$NoEnhancedSell,
    [int]$MaxWorkers = 0,
    [string]$ConfigPath,
    [Nullable[Int32]]$SellAfterDays,
    [Nullable[Int32]]$MinHoldDays,
    [Nullable[Double]]$TakeProfitPct,
    [Nullable[Double]]$StopLossPct,
    [Nullable[Double]]$TrailingStopPct,
    [Nullable[Double]]$PartialSellRatio,
    [switch]$PrioritizeTimeExit,
    [switch]$NoPrioritizeTimeExit
)

# Example usage:
# .\run-strategy01.ps1 `
#   -EnvPath C:\Users\byeun\workspace\perfectdays\.env `
#   -OutputDir C:\Users\byeun\Downloads `
#   -Tickers A010420,A005930 `
#   -MaxWorkers 4
#   -ConfigPath C:\Users\byeun\workspace\perfectdays\config\strategy01.yaml
#   # When omitted, EnvPath/OutputDir/Tickers/MaxWorkers come from strategy01.yaml

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir

if ($EnhancedSell.IsPresent -and $NoEnhancedSell.IsPresent) {
    throw "Specify only one of -EnhancedSell or -NoEnhancedSell."
}

if ($PrioritizeTimeExit.IsPresent -and $NoPrioritizeTimeExit.IsPresent) {
    throw "Specify only one of -PrioritizeTimeExit or -NoPrioritizeTimeExit."
}

if (-not $PSBoundParameters.ContainsKey("ConfigPath")) {
    $ConfigPath = Join-Path (Join-Path $repoRoot "config") "strategy01.yaml"
}

$arguments = @(
    "-m", "backtests.strategy01.eda_strategy01_agent",
    "--config", $ConfigPath,
    "--minimum-records", $MinimumRecords,
    "--min-agent-days", $MinAgentDays
)

if ($PSBoundParameters.ContainsKey("EnvPath")) {
    $arguments += @("--env-path", $EnvPath)
}

if ($PSBoundParameters.ContainsKey("OutputDir")) {
    $arguments += @("--output-dir", $OutputDir)
}

if ($PSBoundParameters.ContainsKey("Budget")) {
    $arguments += @("--budget", $Budget)
}

if ($PSBoundParameters.ContainsKey("ZVol")) {
    $arguments += @("--z-vol", $ZVol)
}

if ($PSBoundParameters.ContainsKey("VwapRelStdMax")) {
    $arguments += @("--vwap-rel-std-max", $VwapRelStdMax)
}

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

if ($NoEnhancedSell.IsPresent) {
    $arguments += "--no-enhanced-sell"
}

if ($PSBoundParameters.ContainsKey("SellAfterDays")) {
    $arguments += @("--sell-after-days", $SellAfterDays)
}

if ($PSBoundParameters.ContainsKey("MinHoldDays")) {
    $arguments += @("--min-hold-days", $MinHoldDays)
}

if ($PSBoundParameters.ContainsKey("TakeProfitPct")) {
    $arguments += @("--take-profit-pct", $TakeProfitPct)
}

if ($PSBoundParameters.ContainsKey("StopLossPct")) {
    $arguments += @("--stop-loss-pct", $StopLossPct)
}

if ($PSBoundParameters.ContainsKey("TrailingStopPct")) {
    $arguments += @("--trailing-stop-pct", $TrailingStopPct)
}

if ($PSBoundParameters.ContainsKey("PartialSellRatio")) {
    $arguments += @("--partial-sell-ratio", $PartialSellRatio)
}

if ($PrioritizeTimeExit.IsPresent) {
    $arguments += "--prioritize-time-exit"
}

if ($NoPrioritizeTimeExit.IsPresent) {
    $arguments += "--no-prioritize-time-exit"
}

if ($MaxWorkers -gt 0) {
    $arguments += @("--max-workers", $MaxWorkers)
}

Push-Location $repoRoot
try {
    Write-Verbose "Running: $Python $($arguments -join ' ')"
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Python @arguments 2>&1 | ForEach-Object { Write-Output $_ }
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousPreference
    if ($exitCode -ne 0) {
        throw "Python exited with code $exitCode"
    }
}
finally {
    Pop-Location
}
