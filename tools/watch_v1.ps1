# watch_v1.ps1
# Keeps this PC awake, prints a run's training status at a steady interval, and restarts training
# if it died unexpectedly. Run it in its own PowerShell window and leave it open:
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\watch_v1.ps1 -Run v2b
#   ... -Run v2      watch (and restart) v2 instead of v1 (or -Run v2b)
#   ... -Run expert-full-house / early-s3 / moe / moe2   the strategy experts, early-game models and joint
#       run (-Stake / -Back for a run at another stake or deck)
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
    [string]$Run = "v1",
    [int]$Stake = 1,
    [string]$Back = "b_red",
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
$hoursGiven = $PSBoundParameters.ContainsKey("Hours")
function Hours-For([int]$d) { if ($hoursGiven) { return $Hours } else { return $d } }
$Slugs = @{ "pair" = "Pair"; "two-pair" = "Two Pair"; "three-of-a-kind" = "Three of a Kind"; "straight" = "Straight";
            "flush" = "Flush"; "full-house" = "Full House"; "four-of-a-kind" = "Four of a Kind"; "straight-flush" = "Straight Flush" }
$base = $Run -replace '(-stake\d+)?(-b_\w+)?$', ''
if ($Run -match '-stake(\d+)') { $Stake = [int]$Matches[1] }      # the run's folder name says its stake and deck
if ($Run -match '-(b_\w+)$') { $Back = $Matches[1] }
$env_ = "export STAKE=$Stake BACK=$Back;"
if (@("v1", "v2", "v2b") -contains $Run) {
    $launch = "cd ~/balatro/balatro-headless && bash train/run_$Run.sh $wslWork $Hours $Workers"
} elseif ($base -match '^expert-(.+)$' -and $Slugs.ContainsKey($Matches[1])) {
    $launch = "cd ~/balatro/balatro-headless && $env_ bash train/run_expert.sh $wslWork '$($Slugs[$Matches[1]])' $(Hours-For 12) $Workers"
} elseif ($base -match '^early-s(\d+)$') {
    $launch = "cd ~/balatro/balatro-headless && $env_ bash train/run_early.sh $wslWork $($Matches[1]) $(Hours-For 8) $Workers"
} elseif ($Run -eq "v3" -or $Run -eq "v3-control") {
    $arm = if ($Run -eq "v3") { "v3" } else { "control" }
    $launch = "cd ~/balatro/balatro-headless && ARMS=$arm bash train/run_v3.sh $wslWork $(Hours-For 168) $Workers"
} elseif ($base -match '^(moe\d*)(-s(\d+))?$') {
    $ms = "export MOE_RUN=$($Matches[1]);" + $(if ($Matches[3]) { " export MOE_SEED=$($Matches[3]);" } else { "" })
    $launch = "cd ~/balatro/balatro-headless && $env_ $ms bash train/run_moe.sh $wslWork $(Hours-For 24) $Workers"
} else {
    Write-Host "don't know how to restart run '$Run'; watching only" -ForegroundColor Yellow
    $NoRestart = $true
    $launch = ""
}
$seenAlerts = @{}
$restarts = 0

function Training-Running {
    # train.py for this run, or any launcher still in its pre-flight checks (they can take minutes
    # before train.py starts; a launcher of another run also counts, which errs toward not restarting)
    & wsl.exe -d $Distro -- pgrep -f "train.py --src.*runs/$Run|train/run_(v1|v2|v2b|moe|expert|early)s?\.sh|tests/(moe|pin)_check\.py|diff_check\.py" *> $null
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
