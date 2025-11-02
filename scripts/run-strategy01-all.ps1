$configs = @(
    "C:\Users\by003457\workspace\perfectdays\config\strategy01_aggressive.yaml",
    "C:\Users\by003457\workspace\perfectdays\config\strategy01_conservative.yaml",
    "C:\Users\by003457\workspace\perfectdays\config\strategy01_creative.yaml"
)

foreach ($cfg in $configs) {
    Write-Host "Running config: $cfg" -ForegroundColor Cyan
    .\scripts\run-strategy01.ps1 -ConfigPath $cfg
}