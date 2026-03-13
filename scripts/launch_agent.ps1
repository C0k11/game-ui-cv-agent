param (
    [switch]$NoLLM
    ,[switch]$ForceRestart
    ,[string]$ApiHost = "127.0.0.1"
    ,[int]$ApiPort = 8000
    ,[string]$ModelPath = "D:\Project\ml_cache\models\Qwen2.5-14B-Instruct-AWQ"
    ,[string]$ModelName = "Qwen2.5-14B-Instruct-AWQ"
    ,[double]$GpuMemoryUtilization = 0.45
    ,[int]$MaxModelLen = 8192
    ,[switch]$UseWslVllm
    ,[string]$WslDistro = ""
    ,[switch]$DryRun
    ,[string[]]$Od = @("Start","Skip","Next","Battle","Event")
    ,[int]$Steps = 0
)

try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
try { chcp 65001 > $null } catch {}

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$logDir = Join-Path $repo "logs"
try { New-Item -ItemType Directory -Force -Path $logDir | Out-Null } catch {}

function Test-TcpPort {
    param([string]$TargetHost, [int]$Port)
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $iar = $client.BeginConnect($TargetHost, $Port, $null, $null)
        $ok = $iar.AsyncWaitHandle.WaitOne(600)
        if (-not $ok) { try { $client.Close() } catch {}; return $false }
        $client.EndConnect($iar)
        try { $client.Close() } catch {}
        return $true
    } catch { return $false }
}

function Wait-TcpPort {
    param([string]$TargetHost, [int]$Port, [int]$TimeoutSec = 30)
    $t0 = Get-Date
    while (((Get-Date) - $t0).TotalSeconds -lt $TimeoutSec) {
        if (Test-TcpPort -TargetHost $TargetHost -Port $Port) { return $true }
        Start-Sleep -Milliseconds 300
    }
    return $false
}

function Stop-ProcessByPort {
    param([int]$Port)
    try {
        $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        foreach ($c in $conns) {
            $procId = $c.OwningProcess
            if ($procId -and $procId -gt 0) {
                try { Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue } catch {}
            }
        }
    } catch {}
}

$processes = @()

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  私人碧蓝档案助手 - Launcher" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

if ($ForceRestart) {
    Stop-ProcessByPort -Port $ApiPort
    Start-Sleep -Milliseconds 400
}

if (-not $NoLLM) {
    if (Test-TcpPort -TargetHost $ApiHost -Port $ApiPort) {
        Write-Host "[0/2] LLM already running on $ApiHost:$ApiPort" -ForegroundColor Gray
    } else {
        if (-not $UseWslVllm) {
            Write-Host "[0/2] LLM not running and --UseWslVllm not set. Start your vLLM server first." -ForegroundColor DarkYellow
        } else {
            $modelPathWsl = $ModelPath
            if ($modelPathWsl -match '^[A-Za-z]:\\') {
                $drive = $modelPathWsl.Substring(0,1).ToLower()
                $rest = $modelPathWsl.Substring(2).Replace('\\','/')
                $modelPathWsl = "/mnt/$drive$rest"
            }

            $cmd = "python -m vllm.entrypoints.openai.api_server --model '$modelPathWsl' --served-model-name '$ModelName' --host 0.0.0.0 --port $ApiPort --gpu-memory-utilization $GpuMemoryUtilization --max-model-len $MaxModelLen"
            if ($WslDistro) {
                $wslArgs = @("-d", $WslDistro, "bash", "-lc", $cmd)
            } else {
                $wslArgs = @("bash", "-lc", $cmd)
            }

            Write-Host "[0/2] Starting vLLM in WSL..." -ForegroundColor Yellow
            Write-Host "      model=$ModelPathWsl" -ForegroundColor DarkGray
            Write-Host "      url=http://$ApiHost:$ApiPort" -ForegroundColor DarkGray

            $out = Join-Path $logDir "vllm.out.log"
            $err = Join-Path $logDir "vllm.err.log"

            if ($DryRun) {
                Write-Host "DRYRUN: wsl $($wslArgs -join ' ')" -ForegroundColor DarkGray
            } else {
                $p = Start-Process -FilePath "wsl" -ArgumentList $wslArgs -PassThru -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err
                $processes += $p
                if (-not (Wait-TcpPort -TargetHost $ApiHost -Port $ApiPort -TimeoutSec 90)) {
                    Write-Host "      -> vLLM did not open port $ApiPort (see logs\vllm.*.log)" -ForegroundColor Red
                } else {
                    Write-Host "      -> vLLM ready" -ForegroundColor Green
                }
            }
        }
    }
} else {
    Write-Host "[0/2] LLM skipped (--NoLLM)" -ForegroundColor DarkGray
}

Write-Host "[1/2] Starting agent loop..." -ForegroundColor Green

$agentOut = Join-Path $logDir "agent.out.log"
$agentErr = Join-Path $logDir "agent.err.log"

$odArgs = @()
foreach ($q in $Od) {
    if ($q) { $odArgs += @("--od", $q) }
}

Write-Host "This launcher no longer starts main.py (legacy ADB agent)." -ForegroundColor Yellow
Write-Host "Use the desktop EXE or start backend via scripts/run_backend.py, then call POST /api/v1/start from dashboard.html." -ForegroundColor Yellow
Write-Host "Reason: main.py requires ADB and is not used in the Windows VLM policy workflow." -ForegroundColor DarkGray
exit 0

try {
    foreach ($proc in $processes) {
        if ($proc -and -not $proc.HasExited) {
            try { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue } catch {}
        }
    }
} catch {}
