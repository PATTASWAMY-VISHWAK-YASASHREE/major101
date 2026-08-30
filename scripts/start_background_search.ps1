param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$Python = "python",
    [string]$OutputDir = "outputs/search/ba75"
)

$absoluteOutput = Join-Path $RepoRoot $OutputDir
New-Item -ItemType Directory -Force -Path $absoluteOutput | Out-Null
$stdout = Join-Path $absoluteOutput "launcher.stdout.log"
$stderr = Join-Path $absoluteOutput "launcher.stderr.log"
$arguments = @(
    "scripts/background_search.py",
    "--output-dir", $OutputDir,
    "--max-hours", "30",
    "--max-attempts", "100",
    "--resume"
)
$process = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $RepoRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
Write-Output ("Started background search PID {0}" -f $process.Id)
