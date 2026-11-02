# lock_balatro.ps1
# Stops Steam from updating Balatro, so the game (and Lovely's patched code) stays pinned for the
# whole dataset. Mods keep working: Lovely loads from the game folder however the game is launched.
#
#   1. Shuts Steam down (it rewrites the manifest while running)
#   2. Sets Balatro to "Only update this game when I launch it" (AutoUpdateBehavior 1)
#   3. Marks appmanifest_2379780.acf read-only, so Steam can't record an update
#   4. Puts a "Balatro (pinned)" shortcut on your Desktop that starts Balatro.exe directly
#      (Steam's Play button will try to update and fail while the lock is on)
#   5. Backs up the game folder and your Mods folder once, to Documents\Balatro_pinned_backup
#   6. Checks Balatro.exe against the pinned SHA-256
#
# Run: right-click -> Run with PowerShell
#   or: powershell -NoProfile -ExecutionPolicy Bypass -File lock_balatro.ps1
# Undo: rerun with -Unlock (clears read-only; Steam can update again)
# Options: -NoBackup   skip the backup copy
#          -NoShortcut skip the Desktop shortcut

param(
    [switch]$Unlock,
    [switch]$NoBackup,
    [switch]$NoShortcut
)

$ErrorActionPreference = "Stop"
$AppId = "2379780"
$PinnedHash = "0D75FE164ACCF3312734D4B37AC98788DD15F0B8E4F9BB8B7F90C4E59DE93F47"   # 1.0.1o-FULL

function Get-SteamPath {
    $reg = Get-ItemProperty -Path "HKCU:\Software\Valve\Steam" -Name SteamPath -ErrorAction SilentlyContinue
    if ($reg -and $reg.SteamPath) { return ($reg.SteamPath -replace '/', '\') }
    foreach ($p in "C:\Program Files (x86)\Steam", "C:\Program Files\Steam") { if (Test-Path $p) { return $p } }
    return $null
}

# The manifest lives in whichever Steam library holds the game.
function Find-Manifest([string]$steam) {
    $libs = @($steam)
    $vdf = Join-Path $steam "steamapps\libraryfolders.vdf"
    if (Test-Path $vdf) {
        foreach ($m in [regex]::Matches((Get-Content $vdf -Raw), '"path"\s+"([^"]+)"')) {
            $libs += ($m.Groups[1].Value -replace '\\\\', '\')
        }
    }
    foreach ($lib in ($libs | Select-Object -Unique)) {
        $acf = Join-Path $lib "steamapps\appmanifest_$AppId.acf"
        if (Test-Path $acf) { return @{ acf = $acf; game = (Join-Path $lib "steamapps\common\Balatro") } }
    }
    return $null
}

function Stop-Steam([string]$steam) {
    if (-not (Get-Process steam -ErrorAction SilentlyContinue)) { return }
    Write-Host "Shutting Steam down..."
    Start-Process (Join-Path $steam "steam.exe") -ArgumentList "-shutdown"
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Seconds 1
        if (-not (Get-Process steam -ErrorAction SilentlyContinue)) { Write-Host "Steam closed."; return }
    }
    throw "Steam is still running after 60 s. Quit it from the tray icon (right-click > Exit) and rerun."
}

try {
    $steam = Get-SteamPath
    if (-not $steam) { throw "Couldn't find Steam." }
    $found = Find-Manifest $steam
    if (-not $found) { throw "Couldn't find appmanifest_$AppId.acf in any Steam library. Is Balatro installed?" }
    $acf, $gameDir = $found.acf, $found.game
    Write-Host "Manifest: $acf"

    if (Get-Process Balatro -ErrorAction SilentlyContinue) { throw "Balatro is running. Close it and rerun." }
    Stop-Steam $steam
    $item = Get-Item $acf

    if ($Unlock) {
        $item.IsReadOnly = $false
        Write-Host ""
        Write-Host "Unlocked. Steam can update Balatro again." -ForegroundColor Yellow
        return
    }

    # 2. Only update on launch. Steam writes this file as UTF-8 without a BOM; keep it that way.
    if ($item.IsReadOnly) { $item.IsReadOnly = $false }
    $text = [System.IO.File]::ReadAllText($acf)
    if ($text -match '"AutoUpdateBehavior"\s+"\d+"') {
        $text = [regex]::Replace($text, '("AutoUpdateBehavior"\s+)"\d+"', '$1"1"')
        [System.IO.File]::WriteAllText($acf, $text, (New-Object System.Text.UTF8Encoding($false)))
        Write-Host "Update setting: only when launched"
    } else {
        Write-Host "AutoUpdateBehavior not in the manifest; skipping (the read-only lock is what matters)" -ForegroundColor Yellow
    }

    # 3. The lock
    (Get-Item $acf).IsReadOnly = $true
    if (-not (Get-Item $acf).IsReadOnly) { throw "Couldn't set the manifest read-only. Rerun as administrator." }
    Write-Host "Manifest: read-only (locked)" -ForegroundColor Green

    # 6. Version check
    $exe = Join-Path $gameDir "Balatro.exe"
    $hash = (Get-FileHash $exe -Algorithm SHA256).Hash
    if ($hash -eq $PinnedHash) {
        Write-Host "Balatro.exe: matches the pinned 1.0.1o build" -ForegroundColor Green
    } else {
        Write-Host "Balatro.exe: DOES NOT match the pinned build ($hash). The game already updated; tell Claude before logging runs." -ForegroundColor Red
    }

    # 4. Shortcut that bypasses Steam's Play button
    if (-not $NoShortcut) {
        $lnk = Join-Path ([Environment]::GetFolderPath("Desktop")) "Balatro (pinned).lnk"
        $sh = New-Object -ComObject WScript.Shell
        $s = $sh.CreateShortcut($lnk)
        $s.TargetPath = $exe
        $s.WorkingDirectory = $gameDir
        $s.Save()
        Write-Host "Shortcut: $lnk"
    }

    # 5. One-time backup
    if (-not $NoBackup) {
        $bk = Join-Path ([Environment]::GetFolderPath("MyDocuments")) "Balatro_pinned_backup"
        if (Test-Path $bk) {
            Write-Host "Backup: already exists at $bk (left as is)"
        } else {
            New-Item -ItemType Directory -Path $bk | Out-Null
            Copy-Item $gameDir (Join-Path $bk "game") -Recurse
            Copy-Item $acf $bk
            $mods = Join-Path $env:APPDATA "Balatro\Mods"
            if (Test-Path $mods) { Copy-Item $mods (Join-Path $bk "Mods") -Recurse }
            Write-Host "Backup: $bk"
        }
    }

    Write-Host ""
    Write-Host "Done. Launch with the 'Balatro (pinned)' shortcut, not Steam's Play button." -ForegroundColor Green
    Write-Host "Steam can be running while you play; it just can't update Balatro."
}
catch {
    Write-Host ""
    Write-Host "Failed: $($_.Exception.Message)" -ForegroundColor Red
}
finally {
    if ($Host.Name -eq "ConsoleHost") { Read-Host "Press Enter to close" | Out-Null }
}
