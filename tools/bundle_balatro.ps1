# bundle_balatro.ps1
# Packages what the headless-sim work needs into one zip on your Desktop:
#   source/   Balatro's Lua source pulled out of Balatro.exe (resources/ art, sound, fonts, shaders skipped)
#   logs/     your newest Antelytics run logs (*.json.gz)
#   mods/     your Mods folder's code: Steamodded, Antelytics, and Lovely's dump of the patched game
#             source (art, sound, fonts and run logs skipped)
#   info.txt  game version, SHA-256 of Balatro.exe (the version pin), Antelytics version, Lovely check
#
# Run: right-click -> Run with PowerShell
#   or: powershell -NoProfile -ExecutionPolicy Bypass -File bundle_balatro.ps1
# Options: -Logs 5            how many recent logs to include (default 3)
#          -GameDir "D:\..."  Balatro install folder, if auto-detect misses it
#
# The bundle contains LocalThunk's code. Upload it to your session; never commit it anywhere public.

param(
    [string]$GameDir = "",
    [string]$DataDir = "",
    [int]$Logs = 3,
    [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

function Find-BalatroDir {
    $libs = @()
    $reg = Get-ItemProperty -Path "HKCU:\Software\Valve\Steam" -Name SteamPath -ErrorAction SilentlyContinue
    if ($reg -and $reg.SteamPath) {
        $steam = $reg.SteamPath -replace '/', '\'
        $libs += $steam
        $vdf = Join-Path $steam "steamapps\libraryfolders.vdf"
        if (Test-Path $vdf) {
            foreach ($m in [regex]::Matches((Get-Content $vdf -Raw), '"path"\s+"([^"]+)"')) {
                $libs += ($m.Groups[1].Value -replace '\\\\', '\')
            }
        }
    }
    $libs += "C:\Program Files (x86)\Steam", "C:\Program Files\Steam"
    foreach ($lib in ($libs | Select-Object -Unique)) {
        $d = Join-Path $lib "steamapps\common\Balatro"
        if (Test-Path (Join-Path $d "Balatro.exe")) { return $d }
    }
    return $null
}

# Balatro.exe is the LOVE runtime with the game's zip appended. Locate the zip via its
# end-of-central-directory record and write just that part out as a normal .zip.
function Export-ZipPayload([string]$exe, [string]$outZip) {
    $bytes = [System.IO.File]::ReadAllBytes($exe)
    $eocd = -1
    $stop = [Math]::Max(0, $bytes.Length - 65557)
    for ($i = $bytes.Length - 22; $i -ge $stop; $i--) {
        if ($bytes[$i] -eq 0x50 -and $bytes[$i+1] -eq 0x4B -and $bytes[$i+2] -eq 0x05 -and $bytes[$i+3] -eq 0x06) {
            $eocd = $i; break
        }
    }
    if ($eocd -lt 0) { throw "No zip archive found inside $exe" }
    $cdSize   = [int64][BitConverter]::ToUInt32($bytes, $eocd + 12)
    $cdOffset = [int64][BitConverter]::ToUInt32($bytes, $eocd + 16)
    $prefix   = [int64]$eocd - $cdSize - $cdOffset
    if ($prefix -lt 0) { throw "Zip layout inside $exe not recognized" }
    $fs = [System.IO.File]::Create($outZip)
    try { $fs.Write($bytes, [int]$prefix, [int]($bytes.Length - $prefix)) } finally { $fs.Close() }
}

function Read-EntryText($entry) {
    $r = New-Object System.IO.StreamReader($entry.Open())
    try { return $r.ReadToEnd() } finally { $r.Close() }
}

function Add-TextEntry($zip, [string]$name, [string]$text) {
    $w = New-Object System.IO.StreamWriter($zip.CreateEntry($name).Open())
    try { $w.Write($text) } finally { $w.Close() }
}

$work = $null; $src = $null; $dst = $null
try {
    if (-not $GameDir) { $GameDir = Find-BalatroDir }
    if (-not $GameDir -or -not (Test-Path (Join-Path $GameDir "Balatro.exe"))) {
        throw "Couldn't find Balatro.exe. In Steam: right-click Balatro > Manage > Browse local files, copy that folder's path, and rerun with -GameDir `"<path>`"."
    }
    if (-not $DataDir) { $DataDir = Join-Path $env:APPDATA "Balatro" }
    if (-not $OutDir)  { $OutDir  = [Environment]::GetFolderPath("Desktop") }
    $exe = Join-Path $GameDir "Balatro.exe"
    Write-Host "Game:  $exe"

    $work = Join-Path ([System.IO.Path]::GetTempPath()) ("balatro_bundle_" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $work | Out-Null
    $payload = Join-Path $work "payload.zip"
    Export-ZipPayload $exe $payload

    $out = Join-Path $OutDir "balatro_bundle.zip"
    if (Test-Path $out) { Remove-Item $out -Force }
    $src = [System.IO.Compression.ZipFile]::OpenRead($payload)
    $dst = [System.IO.Compression.ZipFile]::Open($out, [System.IO.Compression.ZipArchiveMode]::Create)

    # Source files
    $count = 0; $skipped = 0; $version = ""
    foreach ($e in $src.Entries) {
        if ($e.FullName.EndsWith("/")) { continue }
        if ($e.FullName -match '^resources/') { $skipped++; continue }
        if ($e.FullName -eq "version.jkr") { $version = (Read-EntryText $e).Trim() }
        if (-not $version -and $e.FullName -eq "globals.lua") {
            $m = [regex]::Match((Read-EntryText $e), "VERSION\s*=\s*'([^']+)'")
            if ($m.Success) { $version = $m.Groups[1].Value }
        }
        $ne = $dst.CreateEntry("source/" + $e.FullName, [System.IO.Compression.CompressionLevel]::Optimal)
        $in = $e.Open(); $o = $ne.Open()
        try { $in.CopyTo($o) } finally { $o.Close(); $in.Close() }
        $count++
    }
    if ($count -eq 0) { throw "Found the zip inside Balatro.exe but no source files in it." }
    Write-Host "Source: $count files ($skipped art/sound files skipped)"

    # Antelytics version
    $antVersion = "not found"
    $modsDir = Join-Path $DataDir "Mods"
    if (Test-Path $modsDir) {
        $antDir = Get-ChildItem $modsDir -Directory | Where-Object { $_.Name -match 'antelytics' } | Select-Object -First 1
        if ($antDir) {
            $antVersion = "folder found ($($antDir.Name)), version field not found"
            foreach ($j in (Get-ChildItem $antDir.FullName -Filter *.json -Recurse -ErrorAction SilentlyContinue)) {
                $m = [regex]::Match((Get-Content $j.FullName -Raw), '"version"\s*:\s*"([^"]+)"')
                if ($m.Success) { $antVersion = "$($m.Groups[1].Value)  ($($j.Name))"; break }
            }
        }
    }

    # Newest run logs
    $logFiles = @()
    if (Test-Path $DataDir) {
        $logFiles = @(Get-ChildItem $DataDir -Recurse -Filter *.json.gz -File -ErrorAction SilentlyContinue |
                      Sort-Object LastWriteTime -Descending | Select-Object -First $Logs)
    }
    foreach ($f in $logFiles) {
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($dst, $f.FullName, "logs/" + $f.Name) | Out-Null
    }
    Write-Host "Logs:  $($logFiles.Count) included"

    # Mod code. The sim has to run the same patched game you play, so it needs Steamodded
    # and Lovely's dump of the patched source (Mods\lovely\dump).
    $modCount = 0
    if (Test-Path $modsDir) {
        $skipExt = @('.png', '.jpg', '.jpeg', '.ogg', '.wav', '.mp3', '.ttf', '.otf', '.gz', '.zip', '.dll', '.exe')
        $root = (Resolve-Path $modsDir).Path.TrimEnd('\', '/')
        foreach ($f in (Get-ChildItem $modsDir -Recurse -File -ErrorAction SilentlyContinue)) {
            if ($skipExt -contains $f.Extension.ToLower()) { continue }
            $rel = $f.FullName.Substring($root.Length).TrimStart('\', '/') -replace '\\', '/'
            if ($rel -match '(^|/)\.git/') { continue }
            [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($dst, $f.FullName, "mods/" + $rel) | Out-Null
            $modCount++
        }
    }
    Write-Host "Mods:  $modCount files"

    # Info
    $hash = (Get-FileHash $exe -Algorithm SHA256).Hash
    $lovely = (Test-Path (Join-Path $GameDir "version.dll")) -or (Test-Path (Join-Path $GameDir "dwmapi.dll")) -or (Test-Path (Join-Path $GameDir "winmm.dll"))
    $lines = @(
        "created:            $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')",
        "game dir:           $GameDir",
        "game version:       $(if ($version) { $version } else { 'not found in source' })",
        "Balatro.exe sha256: $hash",
        "Balatro.exe bytes:  $((Get-Item $exe).Length)",
        "Lovely installed:   $lovely",
        "Antelytics:         $antVersion",
        "mod files:          $modCount",
        "lovely dump:        $(Test-Path (Join-Path $modsDir 'lovely\dump'))",
        "source files:       $count",
        "logs:"
    )
    foreach ($f in $logFiles) { $lines += "  $($f.Name)  $($f.LastWriteTime.ToString('yyyy-MM-dd HH:mm'))  $($f.FullName)" }
    if ($logFiles.Count -eq 0) { $lines += "  none found under $DataDir" }
    Add-TextEntry $dst "info.txt" ($lines -join "`r`n")

    $dst.Dispose(); $dst = $null
    $src.Dispose(); $src = $null
    $mb = [Math]::Round((Get-Item $out).Length / 1MB, 1)
    Write-Host ""
    Write-Host "Done: $out ($mb MB)" -ForegroundColor Green
    Write-Host "Game version: $(if ($version) { $version } else { 'see info.txt' })   Antelytics: $antVersion"
    Write-Host "Check the logs listed in info.txt are P2 runs, then attach balatro_bundle.zip in the chat."
}
catch {
    Write-Host ""
    Write-Host "Failed: $($_.Exception.Message)" -ForegroundColor Red
}
finally {
    if ($dst) { $dst.Dispose() }
    if ($src) { $src.Dispose() }
    if ($work -and (Test-Path $work)) { Remove-Item $work -Recurse -Force -ErrorAction SilentlyContinue }
    if ($Host.Name -eq "ConsoleHost" -and -not $env:BUNDLE_NO_PAUSE) { Read-Host "Press Enter to close" | Out-Null }
}
