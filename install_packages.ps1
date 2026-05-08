$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Installer = Join-Path $ScriptDir "install_packages.py"

$Python = Get-Command python -ErrorAction SilentlyContinue
if ($Python) {
    & $Python.Source $Installer @args
    exit $LASTEXITCODE
}

$Python3 = Get-Command python3 -ErrorAction SilentlyContinue
if ($Python3) {
    & $Python3.Source $Installer @args
    exit $LASTEXITCODE
}

$PythonLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($PythonLauncher) {
    & $PythonLauncher.Source -3 $Installer @args
    exit $LASTEXITCODE
}

Write-Error "Python was not found. Install Python 3 or activate a conda/venv environment, then rerun this script."
exit 1
