# sysinfo.ps1
# Records the machine a run was trained on, for reproducibility: hardware, Windows, WSL, Linux,
# GPU driver and CUDA, Python packages, the repo commit and the game's hash. Leaves out anything
# that identifies the machine or person (host name, user name, serial numbers, paths under the
# user's folder). Writes <Work>\sysinfo-<date>.txt and prints it.
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\sysinfo.ps1
param(
    [string]$Work = (Join-Path $env:USERPROFILE "balatro-work"),
    [string]$Distro = "Ubuntu-24.04",
    [string]$GameExe = "C:\Program Files (x86)\Steam\steamapps\common\Balatro\Balatro.exe"
)
$ErrorActionPreference = "Continue"
$env:WSL_UTF8 = "1"
$out = New-Object System.Collections.Generic.List[string]
function Add($k, $v) { $out.Add(("{0}: {1}" -f $k, (($v | Out-String).Trim() -replace "\r?\n", " | "))) }

$out.Add("# Recorded " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss K"))
$cs = Get-CimInstance Win32_ComputerSystem
$bios = Get-CimInstance Win32_BIOS
$os = Get-CimInstance Win32_OperatingSystem
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
Add "machine" ("{0} {1}" -f $cs.Manufacturer, $cs.Model)
Add "bios" ("{0}, {1}" -f $bios.SMBIOSBIOSVersion, $(if ($bios.ReleaseDate) { $bios.ReleaseDate.ToString("yyyy-MM-dd") } else { "?" }))
Add "cpu" ("{0}; {1} cores, {2} threads, max {3} MHz" -f $cpu.Name.Trim(), $cpu.NumberOfCores, $cpu.NumberOfLogicalProcessors, $cpu.MaxClockSpeed)
Add "ram" ("{0:N1} GB" -f ($cs.TotalPhysicalMemory / 1GB))
$mods = Get-CimInstance Win32_PhysicalMemory
Add "ram modules" (($mods | ForEach-Object { "{0:N0} GB @ {1} MT/s" -f ($_.Capacity / 1GB), $_.ConfiguredClockSpeed }) -join ", ")
foreach ($g in Get-CimInstance Win32_VideoController) { Add "gpu (windows)" ("{0}, driver {1} ({2})" -f $g.Name, $g.DriverVersion, $(if ($g.DriverDate) { $g.DriverDate.ToString("yyyy-MM-dd") } else { "?" })) }
Add "windows" ("{0}, version {1}, build {2}.{3}" -f $os.Caption, (Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion").DisplayVersion, $os.BuildNumber, (Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion").UBR)
Add "power plan" ((powercfg /getactivescheme) -replace "^.*\(", "" -replace "\)\s*$", "")
$bat = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue
if ($bat) { Add "on AC power" ($bat.BatteryStatus -eq 2) }
$vbs = Get-CimInstance -ClassName Win32_DeviceGuard -Namespace root\Microsoft\Windows\DeviceGuard -ErrorAction SilentlyContinue
if ($vbs) { Add "memory integrity (HVCI) running" ($vbs.SecurityServicesRunning -contains 2) }
$asus = Get-CimInstance Win32_PnPSignedDriver | Where-Object { $_.Manufacturer -match "ASUS" -and $_.DeviceClass -eq "SYSTEM" }
foreach ($d in $asus) { Add "asus system driver" ("{0} {1}" -f $d.DeviceName, $d.DriverVersion) }
Add "wsl" ((& wsl.exe --version) | Where-Object { $_ -match "\S" })
$wc = Join-Path $env:USERPROFILE ".wslconfig"
Add ".wslconfig" $(if (Test-Path $wc) { (Get-Content $wc | Where-Object { $_ -match "\S" -and $_ -notmatch "^\s*#" }) -join "; " } else { "none (WSL defaults)" })
if (Test-Path $GameExe) { Add "Balatro.exe sha256" (Get-FileHash $GameExe -Algorithm SHA256).Hash } else { Add "Balatro.exe" "not at $GameExe (pass -GameExe)" }

$linux = @'
. /etc/os-release; echo "linux: $PRETTY_NAME, kernel $(uname -r)"
echo "linux cpus: $(nproc); linux ram: $(free -g | awk '/Mem:/{print $2}') GB; swap: $(free -g | awk '/Swap:/{print $2}') GB"
echo "gpu (linux): $(nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv,noheader 2>/dev/null)"
echo "cuda (driver): $(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9.]*' | cut -d' ' -f3)"
. ~/balatro/venv/bin/activate 2>/dev/null
echo "python (venv): $(python3 --version 2>&1 | cut -d' ' -f2)"
python3 - <<'P' 2>&1
import importlib
for m in ("torch", "numpy", "lupa"):
    try:
        x = importlib.import_module(m)
        extra = ""
        if m == "torch":
            extra = f", cuda {x.version.cuda}, cudnn {x.backends.cudnn.version()}, gpu available {x.cuda.is_available()}"
        print(f"{m}: {x.__version__}{extra}")
    except Exception as e:
        print(f"{m}: not importable ({type(e).__name__})")
P
echo "gcc: $(gcc -dumpfullversion 2>/dev/null)"
echo "git: $(git --version | cut -d' ' -f3)"
R=~/balatro/balatro-headless
echo "repo commit: $(git -C $R rev-parse --short HEAD) ($(git -C $R log -1 --format=%cs))"
L=$(find ~/balatro -maxdepth 4 -name 'libluajit*.so*' -o -maxdepth 4 -name 'lua51*.so' 2>/dev/null | head -1)
[ -n "$L" ] && echo "luajit lib sha256: $(sha256sum "$L" | cut -c1-16)…"
'@
$sh = Join-Path $Work "sysinfo_linux.sh"          # a file, not an argument: no quoting through wsl.exe
[IO.File]::WriteAllText($sh, ($linux -replace "`r`n", "`n"))
$wslSh = "/mnt/" + $sh.Substring(0, 1).ToLower() + "/" + ($sh.Substring(3) -replace '\\', '/')
(& wsl.exe -d $Distro -- bash $wslSh) | ForEach-Object { $out.Add($_) }
Remove-Item $sh -ErrorAction SilentlyContinue

$file = Join-Path $Work ("sysinfo-" + (Get-Date -Format "yyyy-MM-dd") + ".txt")
$out | Set-Content -Encoding UTF8 $file
$out
Write-Host "`nwrote $file" -ForegroundColor Green
