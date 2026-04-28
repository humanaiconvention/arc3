param(
    [int]$Port = 8888,
    [string]$Root = "D:\arc3"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $Root)) {
    throw "Root path not found: $Root"
}

$rootPath = (Resolve-Path -LiteralPath $Root).Path
Set-Location -LiteralPath $rootPath

# Colab's local-runtime docs note that some frontend features expect a /content
# directory to exist under the Jupyter working directory.
$contentDir = Join-Path $rootPath "content"
New-Item -ItemType Directory -Force -Path $contentDir | Out-Null

Write-Host "Starting Jupyter for Colab local runtime from $rootPath on port $Port..."
Write-Host "When Jupyter prints a URL like http://127.0.0.1:$Port/?token=..., copy it into Colab via Connect > Connect to local runtime."

python -m notebook `
    --NotebookApp.allow_origin='https://colab.research.google.com' `
    --NotebookApp.port_retries=0 `
    --NotebookApp.allow_credentials=True `
    --port=$Port `
    --no-browser
