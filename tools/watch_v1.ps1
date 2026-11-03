# watch_v1.ps1
# Keeps this PC awake, prints a run's training status at a steady interval, and restarts training
# if it died unexpectedly. Run it in its own PowerShell window and leave it open:
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\watch_v1.ps1 -Run v2b
#   ... -Run v2      watch (and restart) v2 instead of v1 (or -Run v2b)
# Options: -Interval 60       seconds between status lines
#          -StaleMinutes 15   no health update for this long, and no training process = died
#          -NoRestart         only watch, never restart
# Ctrl+C here stops the watcher (and lets the PC sleep again); training is not affected.
#
# Restarts only after an unexpected death: health.json stale with no training process running,
# or a status of "crashed" / "workers failed". Never after a deliberate stop (Ctrl+C in the
# training window, the win target, the time cap, or the not-learning stop).

param(
    [string]$Work = (Join-Path $env:USERPROFILE "balatro-work"),
    [string]$Distro = "Ubuntu-24.04",
    [ValidateSet("v1", "v2", "v2b")][string]$Run = "v1",
    [int]$Interval = 60,
    [int]$StaleMinutes = 15,
    [int]$Hours = 168,
    [int]$Workers = 14,
    [switch]$NoRestart
)

Add-Type -Namespace Power -Name Native -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
'@
$ES_CONTINUOUS = [uint32]"0x80000000"; $ES_SYSTEM_REQUIRED = [uint32]"0x00000001"
function Stay-Awake { [void][Power.Native]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) }
function Allow-Sleep { [void][Power.Native]::SetThreadExecutionState($ES_CONTINUOUS) }

$runDir = Join-Path $Work ("runs\" + $Run)   # not $run: PowerShell names ignore case
$healthPath = Join-Path $runDir "health.json"
$metricsPath = Join-Path $runDir "metrics.jsonl"
$wslWork = "/mnt/c/" + ($Work.Substring(3) -replace '\\', '/')
$launch = "cd ~/balatro/balatro-headless && bash train/run_$Run.sh $wslWork $Hours $Workers"
$seenAlerts = @{}
$restarts = 0

function Training-Running {
    & wsl.exe -d $Distro -- pgrep -f "train.py --src.*runs/$Run" *> $null
    return ($LASTEXITCODE -eq 0)
}

function Start-Training($why) {
    $script:restarts++
    Write-Host ("{0}  RESTARTING training ({1}); restart #{2}" -f (Get-Date -Format "HH:mm:ss"), $why, $script:restarts) -ForegroundColor Yellow
    Start-Process -FilePath "wsl.exe" -ArgumentList @("-d", $Distro, "--", "bash", "-c", "`"$launch`"") -WindowStyle Minimized
}

function Train-Average {
    if (-not (Test-Path $metricsPath)) { return $null }
    $vals = @()
    foreach ($line in (Get-Content $metricsPath -Tail 40)) {
        if ($line -match '"train"') {
            try { $vals += ([double](($line | ConvertFrom-Json).train.blinds_mean)) } catch {}
        }
    }
    if ($vals.Count -eq 0) { return $null }
    $last = $vals | Select-Object -Last 10
    return ($last | Measure-Object -Average).Average
}

Write-Host "Watching $runDir every $Interval s. Keeping the PC awake. Ctrl+C to stop watching." -ForegroundColor Cyan
try {
    while ($true) {
        Stay-Awake
        $now = Get-Date
        if (-not (Test-Path $healthPath)) {
            Write-Host ("{0}  no health.json yet" -f $now.ToString("HH:mm:ss"))
        } else {
            try { $h = Get-Content $healthPath -Raw | ConvertFrom-Json } catch { $h = $null }
            if ($h) {
                $updated = [datetime]::ParseExact($h.updated, "yyyy-MM-dd HH:mm:ss", $null)
                $age = ($now - $updated).TotalMinutes
                $avg = Train-Average
                $wins = if ($null -ne $h.wins_total) { "{0:N0}" -f $h.wins_total } else { "?" }
                $line = "{0}  {1}  round {2}  games {3:N0}  wins {4}  train {5} blinds/game  best eval {6}  {7} dec/s  mem {8} GB  updated {9:N0} min ago" -f `
                    $now.ToString("HH:mm:ss"), $h.status, $h.iteration, $h.games_total, $wins,
                    ($(if ($avg) { "{0:N2}" -f $avg } else { "?" })), $h.best_eval_blinds, $h.decisions_per_second,
                    $h.mem_available_gb, $age
                $color = if ($h.status -eq "running") { "Green" } else { "Yellow" }
                Write-Host $line -ForegroundColor $color
                foreach ($a in @($h.alerts)) {
                    if ($a -and -not $seenAlerts.ContainsKey($a)) { $seenAlerts[$a] = $true; Write-Host "   $a" -ForegroundColor Magenta }
                }

                if (-not $NoRestart) {
                    $died = $false; $why = ""
                    if ($h.status -eq "running" -and $age -gt $StaleMinutes -and -not (Training-Running)) {
                        $died = $true; $why = "no update for $([int]$age) min and no training process"
                    } elseif ($h.status -match "^stopped: (crashed|workers failed|workers could not restart)" -and -not (Training-Running)) {
                        $died = $true; $why = $h.status.Substring(0, [Math]::Min(80, $h.status.Length))
                    }
                    if ($died) {
                        if ($restarts -ge 5) {
                            Write-Host "   5 restarts already; not restarting again. Check train.log." -ForegroundColor Red
                        } else {
                            Start-Training $why
                        }
                    }
                }
            }
        }
        Start-Sleep -Seconds $Interval
    }
}
finally {
    Allow-Sleep
    Write-Host "Watcher stopped; the PC may sleep again per its power settings." -ForegroundColor Cyan
}
