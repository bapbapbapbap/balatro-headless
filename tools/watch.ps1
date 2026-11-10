# watch.ps1
# Keeps this PC awake, prints a run's training status at a steady interval, and restarts training
# if it died unexpectedly. Run it in its own PowerShell window and leave it open:
#   powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\<you>\balatro-work\watch.ps1 -Run v4b
# A restart relaunches the run the way the clone's branch launches it: `train/launch.sh <run>` when the
# branch has configs/<run>.toml, else the run's own train/run_<run>.sh (runs from before the launcher).
# Options: -Interval 60       seconds between status lines
#          -StaleMinutes 15   no health update for this long, and no training process = died
#          -Hours / -Workers  passed to the relaunch (default: the run's config)
#          -NoRestart         only watch, never restart
# Ctrl+C here stops the watcher (and lets the PC sleep again); training is not affected.
#
# Restarts only after an unexpected death: health.json stale with no training process running,
# or a status of "crashed" / "workers failed". Never after a deliberate stop (Ctrl+C in the
# training window, the win target, the time cap, a stop rule).

param(
    [string]$Work = (Join-Path $env:USERPROFILE "balatro-work"),
    [string]$Distro = "Ubuntu-24.04",
    [Parameter(Mandatory = $true)][string]$Run,
    [int]$Interval = 60,
    [int]$StaleMinutes = 15,
    [int]$Hours = 0,
    [int]$Workers = 0,
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
$opts = $(if ($Hours -gt 0) { " --hours $Hours" } else { "" }) + $(if ($Workers -gt 0) { " --workers $Workers" } else { "" })
$hw = "$(if ($Hours -gt 0) { $Hours } else { 168 }) $(if ($Workers -gt 0) { $Workers } else { 12 })"
# the clone decides: a config on its branch means the launcher; otherwise the run's own script
$launch = "cd ~/balatro/balatro-headless && if [ -f configs/$Run.toml ]; then bash train/launch.sh $Run --work $wslWork$opts; " +
          "elif [ -f train/run_$Run.sh ]; then bash train/run_$Run.sh $wslWork $hw; else echo no launcher for $Run; exit 1; fi"
$seenAlerts = @{}
$restarts = 0

function Training-Running {
    # train.py for this run, or any launcher still in its pre-flight checks (they can take minutes
    # before train.py starts; a launcher of another run also counts, which errs toward not restarting)
    & wsl.exe -d $Distro -- pgrep -f "train.py --src.*runs/$Run|train/launch\.(sh|py)|train/run_\w+\.sh|tests/\w+_check\.py|diff_check\.py|verify_games\.py" *> $null
    return ($LASTEXITCODE -eq 0)
}

$GraceMinutes = 30          # after a restart, no other restart until training has written health.json or this long has passed
$lastLaunch = $null

function Start-Training($why) {
    $script:restarts++
    Write-Host ("{0}  RESTARTING training ({1}); restart #{2}" -f (Get-Date -Format "HH:mm:ss"), $why, $script:restarts) -ForegroundColor Yellow
    Start-Process -FilePath "wsl.exe" -ArgumentList @("-d", $Distro, "--", "bash", "-c", "`"$launch`"") -WindowStyle Minimized
    $script:lastLaunch = Get-Date
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
                    if ($died -and $lastLaunch -and $updated -lt $lastLaunch -and ($now - $lastLaunch).TotalMinutes -lt $GraceMinutes) {
                        $died = $false      # the last restart is still starting up (pre-flight); give it time
                        Write-Host ("   restarted {0:N0} min ago; waiting for it to start training (up to {1} min)" -f ($now - $lastLaunch).TotalMinutes, $GraceMinutes) -ForegroundColor DarkYellow
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
