$ErrorActionPreference = "Stop"

try {
    $repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
    Set-Location $repoRoot

    # .venv があれば優先使用
    $venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        $python = $venvPython
    } else {
        $python = "python"
    }

    $port = if ($env:KNOWLEDGE_DISTILLATION_PORT) { $env:KNOWLEDGE_DISTILLATION_PORT } else { "8501" }
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"
    $env:STREAMLIT_SERVER_HEADLESS = "true"
    $env:BROWSER = "none"

    $requirementsPath = Join-Path $repoRoot "knowledge_distillation\requirements.txt"

    function Install-PipIfMissing {
        & $python -m pip --version
        if ($LASTEXITCODE -ne 0) {
            Write-Host "pip not found. Installing pip..."
            & $python -m ensurepip --upgrade
            if ($LASTEXITCODE -ne 0) {
                throw "pip のセットアップに失敗しました。"
            }
        }
    }

    function Install-Requirements {
        param(
            [Parameter(Mandatory = $true)]
            [string]$RequirementsFile
        )

        if (-not (Test-Path $RequirementsFile)) {
            throw "requirements.txt が見つかりません: $RequirementsFile"
        }

        Install-PipIfMissing

        Write-Host "Installing dependencies from: $RequirementsFile"
        & $python -m pip install -r $RequirementsFile
        if ($LASTEXITCODE -ne 0) {
            throw "requirements.txt からの依存関係インストールに失敗しました。"
        }
    }

    Write-Host "========================================"
    Write-Host "Repo root        : $repoRoot"
    Write-Host "Python           : $python"
    Write-Host "Requirements path: $requirementsPath"
    Write-Host "Port             : $port"
    Write-Host "========================================"
    Write-Host ""

    & $python --version
    if ($LASTEXITCODE -ne 0) {
        throw "Python の起動に失敗しました。python='$python'"
    }

    $existing = Get-CimInstance Win32_Process |
        Where-Object {
            $_.CommandLine -and
            $_.CommandLine -match "streamlit run knowledge_distillation/app.py" -and
            $_.CommandLine -match "--server.port\s+$port"
        }

    foreach ($process in $existing) {
        Write-Host "Stopping existing Streamlit process on port $port (PID: $($process.ProcessId))"
        Stop-Process -Id $process.ProcessId -Force
    }

    # requirements.txt から依存関係をインストール
    Install-Requirements -RequirementsFile $requirementsPath

    Write-Host ""
    Write-Host "Starting Knowledge Distillation UI..."
    Write-Host "URL: http://localhost:$port"
    Write-Host ""
    Write-Host "Demo import CSV:"
    Write-Host "  benchmark\demo\knowledge_distillation_start_inquiries.csv"
    Write-Host "Comparison source:"
    Write-Host "  data\approved_knowledge.json"
    Write-Host ""
    Write-Host "If Azure OpenAI variables are missing, copy .env.example to .env and set real values."
    Write-Host ""

    Start-Process "http://localhost:$port"
    & $python -m streamlit run knowledge_distillation/app.py --server.port $port --server.headless true --browser.gatherUsageStats false

    exit 0
}
catch {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "ERROR: open_knowledge_distillation.ps1 failed" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "Message:" -ForegroundColor Yellow
    Write-Host $_.Exception.Message
    Write-Host ""
    Write-Host "CategoryInfo:" -ForegroundColor Yellow
    Write-Host $_.CategoryInfo
    Write-Host ""
    Write-Host "ScriptStackTrace:" -ForegroundColor Yellow
    Write-Host $_.ScriptStackTrace
    Write-Host ""

    exit 1
}
