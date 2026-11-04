# balatro.ps1 - one controller for the whole setup, from Windows PowerShell.
#
#   Set-ExecutionPolicy -Scope Process Bypass     (once per window)
#   .\tools\balatro.ps1 <command> [options]
#
# Commands (in setup order):
#   check     Balatro.exe hash vs the pin, Lovely/mods present, WSL distro, bundle, work folder
#   lock      stop Steam updating Balatro (tools\lock_balatro.ps1); "unlock" undoes it
#   bundle    extract the game's Lua source + mods + logs into balatro_bundle.zip in -Work
#   setup     WSL: clone the repo, build LuaJIT 2.0.5 + lupa, unpack the bundle, run every check
#   train     start (or resume) training for -Run in its own window
#   stop      stop -Run cleanly (it saves latest.pt)
#   status    one-screen status of -Run from health.json and the last evals
#   watch     keep the PC awake and restart -Run only if it crashes (tools\watch_v1.ps1)
#   verify    replay every route in wins\routes.jsonl on the untouched game
#   publish   replay every new win from every run under -Work\runs, add it to wins\, update the
#             tally (README badge and counts), commit and push
#
# The mixture of experts and the save (docs/hypotheses.md section I and roadmap; each starts in
# its own window and resumes where it stopped):
#   checks    every deck and challenge starts right; fast sim vs the untouched game on every deck
#             at White and Gold Stake and on every challenge
#   experts   train the hand experts one after another (-Hands "Full House|Flush" for some; -Hours each)
#   earlies   grow early-game models (-Seeds "1 2 3"; -Hours each)
#   moe       joint training from the best early model and the 8 experts (-TrainEarly to train it too)
#   save      play one save from a new profile with -Weights, writing every run to -Save (-Runs N)
#   checksave replay -Save in order on a new profile; says SAVE 100% PROVEN only at 100%
#   progress  the save-progress tables in docs\proof.md from the published wins; commit and push
#   help      this list
#
# Options: -Work C:\Users\<you>\balatro-work   -Run v2b   -Distro Ubuntu-24.04
#          -Workers 14   -Hours 168   -Stake 1   -Back b_red
#          -Hands "..."   -Seeds "..."   -TrainEarly   -Weights <file>   -Save <file>   -Runs 100
#
# Nothing here touches the game's files except "lock" (read-only flag on Steam's manifest, a
# shortcut and a backup; see lock_balatro.ps1). Training, the game source and every result live
# under -Work and inside WSL.

param(
    [Parameter(Position = 0)][string]$Command = "help",
    [string]$Work = (Join-Path $env:USERPROFILE "balatro-work"),
    [string]$Run = "v2b",
    [string]$Distro = "Ubuntu-24.04",
    [int]$Workers = 14,
    [int]$Hours = 168,
    [int]$Stake = 1,
    [string]$Back = "b_red",
    [string]$Hands = "",
    [string]$Seeds = "",
    [switch]$TrainEarly,
    [string]$Weights = "",
    [string]$Save = "",
    [int]$Runs = 100
)

$ErrorActionPreference = "Stop"
$script:HoursGiven = $PSBoundParameters.ContainsKey("Hours")   # experts/earlies/moe have their own default hours
$Tools = $PSScriptRoot
$PinnedHash = "0D75FE164ACCF3312734D4B37AC98788DD15F0B8E4F9BB8B7F90C4E59DE93F47"
$RepoRaw = "https://raw.githubusercontent.com/bapbapbapbap/balatro-headless/main"

function Say($msg, $color = "Gray") { Write-Host $msg -ForegroundColor $color }
function Ok($msg) { Say ("  OK    " + $msg) Green }
function Bad($msg) { Say ("  NO    " + $msg) Red }

function Wsl-Path([string]$winPath) {
    # C:\Users\x\balatro-work -> /mnt/c/Users/x/balatro-work
    $drive = $winPath.Substring(0, 1).ToLower()
    return "/mnt/$drive/" + ($winPath.Substring(3) -replace '\\', '/')
}

function Wsl([string]$bash) {
    & wsl.exe -d $Distro -- bash -lc $bash | Out-Host
    return $LASTEXITCODE
}

function Find-Game {
    $roots = @()
    try {
        $steam = (Get-ItemProperty -Path "HKCU:\Software\Valve\Steam" -ErrorAction Stop).SteamPath
        if ($steam) { $roots += ($steam -replace '/', '\') }
    } catch {}
    $roots += "C:\Program Files (x86)\Steam"
    foreach ($r in $roots) {
        $vdf = Join-Path $r "steamapps\libraryfolders.vdf"
        $libs = @($r)
        if (Test-Path $vdf) {
            foreach ($m in [regex]::Matches((Get-Content $vdf -Raw), '"path"\s+"([^"]+)"')) { $libs += ($m.Groups[1].Value -replace '\\\\', '\') }
        }
        foreach ($l in $libs) {
            $d = Join-Path $l "steamapps\common\Balatro"
            if (Test-Path (Join-Path $d "Balatro.exe")) { return $d }
        }
    }
    return $null
}

function Cmd-Check {
    Say "Checking the setup" White
    $game = Find-Game
    if ($game) {
        $h = (Get-FileHash (Join-Path $game "Balatro.exe") -Algorithm SHA256).Hash
        if ($h -eq $PinnedHash) { Ok "Balatro.exe is 1.0.1o-FULL ($game)" } else { Bad "Balatro.exe hash $h is not the pinned 1.0.1o build; wins\ routes won't replay on it" }
        if (Test-Path (Join-Path $game "version.dll")) { Ok "Lovely installed (version.dll)" } else { Say "  --    Lovely not installed (only needed to log your own games)" }
    } else { Bad "Balatro not found in any Steam library" }
    $mods = Join-Path $env:APPDATA "Balatro\Mods"
    foreach ($m in @("smods", "Antelytics", "lovely\dump")) {
        if (Test-Path (Join-Path $mods $m)) { Ok "Mods\$m" } else { Say "  --    Mods\$m not present (only needed for logging / the modded game)" }
    }
    if (Test-Path $Work) { Ok "work folder $Work" } else { Bad "work folder $Work missing (create it, or pass -Work)" }
    if (Test-Path (Join-Path $Work "balatro_bundle.zip")) { Ok "balatro_bundle.zip in the work folder" } else { Bad "no balatro_bundle.zip in the work folder (run: bundle)" }
    $distros = (& wsl.exe -l -q 2>$null) -replace "`0", "" | Where-Object { $_ -ne "" }
    if ($distros -contains $Distro) { Ok "WSL distro $Distro" } else { Bad "WSL distro $Distro not found (have: $($distros -join ', ')); install it or pass -Distro" }
    if ($distros -contains $Distro) {
        & wsl.exe -d $Distro -- bash -lc "test -x ~/balatro/venv/bin/python3 && test -d ~/balatro/balatro-headless" 2>$null
        if ($LASTEXITCODE -eq 0) { Ok "harness set up in WSL (~/balatro)" } else { Say "  --    harness not set up in WSL yet (run: setup)" }
    }
}

function Cmd-Bundle {
    if (-not (Test-Path $Work)) { New-Item -ItemType Directory -Path $Work | Out-Null }
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Tools "bundle_balatro.ps1") -OutDir $Work
}

function Cmd-Setup {
    $w = Wsl-Path $Work
    Say "Setting up the harness in WSL ($Distro). This asks for your Linux password once (apt)." White
    $bash = "set -e; sudo apt-get update -qq && sudo apt-get install -y -qq build-essential git curl python3-dev python3-venv python3-pip >/dev/null; " +
            "curl -sSL -o /tmp/setup_linux.sh $RepoRaw/tools/setup_linux.sh && bash /tmp/setup_linux.sh '$w'"
    $rc = Wsl $bash
    Say ("setup exited with {0}; report: {1}" -f $rc, (Join-Path $Work "setup_report.txt"))
}

function To-Wsl([string]$p) {
    # a Windows path (C:\...) becomes /mnt/c/...; anything else is taken as a path inside WSL / the repo
    if ($p -match '^[A-Za-z]:\\') { return Wsl-Path $p }
    return $p
}

function Hours-Or([int]$default) {
    if ($script:HoursGiven) { return $Hours }
    return $default
}

function Start-Window([string]$what, [string]$bash) {
    $launch = "cd ~/balatro/balatro-headless && git pull -q --ff-only; export STAKE=$Stake BACK=$Back; $bash"
    Start-Process -FilePath "wsl.exe" -ArgumentList @("-d", $Distro, "--", "bash", "-lc", "`"$launch; exec bash`"")
    Say "started $what in a new window" Green
}

function Cmd-Checks {
    [void](Wsl "cd ~/balatro/balatro-headless && git pull -q --ff-only; . ~/balatro/venv/bin/activate && python3 tests/start_check.py --src ~/balatro/local/source | tail -25 && bash tests/run_coverage_checks.sh ~/balatro/local/source")
}

function Cmd-Experts {
    $w = Wsl-Path $Work
    $h = if ($Hands) { "HANDS='$Hands' " } else { "" }
    Start-Window "the experts" "${h}bash train/run_experts.sh '$w' $(Hours-Or 12) $Workers"
}

function Cmd-Earlies {
    $w = Wsl-Path $Work
    $s = if ($Seeds) { "SEEDS='$Seeds' " } else { "" }
    Start-Window "the early-game models" "${s}bash train/run_earlies.sh '$w' $(Hours-Or 8) $Workers"
}

function Cmd-Moe {
    $w = Wsl-Path $Work
    $t = if ($TrainEarly) { "MOE_TRAIN_EARLY=1 " } else { "" }
    Start-Window "joint training (moe)" "${t}bash train/run_moe.sh '$w' $(Hours-Or 24) $Workers"
}

function Save-Path {
    if ($Save) { return To-Wsl $Save }
    return (Wsl-Path $Work) + "/saves/save1.jsonl"
}

function Cmd-Save {
    $wt = if ($Weights) { To-Wsl $Weights } else { (Wsl-Path $Work) + "/runs/moe/best.pt" }
    Start-Window "the save ($(Save-Path))" (". ~/balatro/venv/bin/activate && python3 tools/play_save.py --src ~/balatro/local/source " +
        "--weights '$wt' --save '$(Save-Path)' --runs $Runs")
}

function Cmd-CheckSave {
    [void](Wsl "cd ~/balatro/balatro-headless && git pull -q --ff-only; . ~/balatro/venv/bin/activate && python3 train/verify_games.py --src ~/balatro/local/source --save '$(Save-Path)' | tail -8")
}

function Cmd-Progress {
    $bash = "cd ~/balatro/balatro-headless && git pull -q --ff-only && . ~/balatro/venv/bin/activate && " +
            "python3 tools/save_progress.py --src ~/balatro/local/source --jobs $Workers && " +
            "git add docs/proof.md wins/won_with.jsonl wins/save_progress.json && " +
            "if git diff --cached --quiet; then echo 'nothing new'; else git commit -q -m 'proof: save-progress tables from the game record' && " +
            "{ git push -q && echo 'pushed'; } || echo 'committed but not pushed: run git push in ~/balatro/balatro-headless'; fi"
    [void](Wsl $bash)
}

function Cmd-Train {
    if (@("v1", "v2", "v2b") -notcontains $Run) { Bad "train starts v1, v2 or v2b; for the others use experts, earlies or moe"; return }
    $w = Wsl-Path $Work
    $script = "train/run_$Run.sh"
    if ($Run -eq "v2b" -and -not (Test-Path (Join-Path $Work "runs\v2b\latest.pt"))) {
        $script = "tools/start_v2b.sh"     # first v2b launch: stops v2 cleanly, starts from its latest.pt
    }
    $launch = "cd ~/balatro/balatro-headless && git pull -q --ff-only; bash $script $w $Hours $Workers"
    Start-Process -FilePath "wsl.exe" -ArgumentList @("-d", $Distro, "--", "bash", "-lc", "`"$launch; exec bash`"")
    Say "started $Run in a new window ($script). Watch it with: status, or watch" Green
}

function Cmd-Stop {
    [void](Wsl "pkill -INT -f 'train.py .*--run $Run ' && echo 'stop requested; it saves latest.pt and exits within seconds' || echo 'no $Run training process found'")
}

function Cmd-Status {
    $dir = Join-Path $Work "runs\$Run"
    $hp = Join-Path $dir "health.json"
    if (-not (Test-Path $hp)) { Bad "no $hp yet"; return }
    $h = Get-Content $hp -Raw | ConvertFrom-Json
    $age = [int]((Get-Date) - (Get-Item $hp).LastWriteTime).TotalMinutes
    Say ("{0}: {1} (updated {2}, {3} min ago)" -f $Run, $h.status, $h.updated, $age) White
    Say ("  round {0}, {1:N0} games, {2:N0} wins, {3:N0} beaten seeds" -f $h.iteration, $h.games_total, $h.wins_total, $h.beaten_seeds)
    if ($h.last_eval) { Say ("  last eval {0:N2} blinds/game, {1} wins in {2}; best {3:N3}" -f $h.last_eval.blinds_mean, $h.last_eval.wins, $h.last_eval.games, $h.best_eval_blinds) }
    Say ("  lr {0}, {1:N1} GB free" -f $h.lr, $h.mem_available_gb)
    $log = Join-Path $dir "train.log"
    if (Test-Path $log) {
        Say "  last rounds:"
        Get-Content $log -Tail 200 | Where-Object { $_ -match '^\[it ' } | Select-Object -Last 3 | ForEach-Object { Say ("    " + $_.Substring(0, [Math]::Min(200, $_.Length))) }
        Get-Content $log -Tail 400 | Where-Object { $_ -match 'ALERT' } | Select-Object -Last 3 | ForEach-Object { Say ("    " + $_) Yellow }
    }
}

function Cmd-Watch {
    $a = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $Tools "watch_v1.ps1"), "-Run", $Run, "-Work", $Work,
           "-Distro", $Distro, "-Workers", $Workers, "-Stake", $Stake, "-Back", $Back)
    if ($script:HoursGiven) { $a += @("-Hours", $Hours) }
    & powershell @a
}

function Cmd-Verify {
    [void](Wsl "cd ~/balatro/balatro-headless && git pull -q --ff-only; . ~/balatro/venv/bin/activate && cd train && python3 verify_games.py --src ~/balatro/local/source ../wins/routes.jsonl --wins-only --full | tail -3")
}

function Cmd-Publish {
    $w = Wsl-Path $Work
    $bash = "cd ~/balatro/balatro-headless && git pull -q --ff-only && . ~/balatro/venv/bin/activate && cd train && " +
            "{ python3 publish_wins.py --src ~/balatro/local/source --out ../wins '$w'/runs/*/wins.jsonl || echo 'some wins did not replay exactly and were left out (listed above)'; } && " +
            "python3 tally.py && cd .. && git add wins README.md && " +
            "if git diff --cached --quiet; then echo 'nothing new to publish'; " +
            "else git commit -q -m 'wins: publish new beaten seeds and update the tally' && " +
            "{ git push -q && echo 'pushed'; } || echo 'committed but not pushed: run git push in ~/balatro/balatro-headless'; fi"
    [void](Wsl $bash)
}

switch ($Command.ToLower()) {
    "check"  { Cmd-Check }
    "lock"   { & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Tools "lock_balatro.ps1") }
    "unlock" { & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Tools "lock_balatro.ps1") -Unlock }
    "bundle" { Cmd-Bundle }
    "setup"  { Cmd-Setup }
    "train"  { Cmd-Train }
    "stop"   { Cmd-Stop }
    "status" { Cmd-Status }
    "watch"  { Cmd-Watch }
    "verify" { Cmd-Verify }
    "publish" { Cmd-Publish }
    "checks"  { Cmd-Checks }
    "experts" { Cmd-Experts }
    "earlies" { Cmd-Earlies }
    "moe"     { Cmd-Moe }
    "save"    { Cmd-Save }
    "checksave" { Cmd-CheckSave }
    "progress" { Cmd-Progress }
    default  { Get-Content $PSCommandPath | Select-Object -Skip 1 -First 36 | ForEach-Object { $_ -replace '^# ?', '' } }
}
