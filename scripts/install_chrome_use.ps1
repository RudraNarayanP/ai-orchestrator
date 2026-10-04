<#
.SYNOPSIS
  Installs the chrome-use CLI (Windows x64) from its official GitHub release, with a checksum check.

.DESCRIPTION
  OmniBrain's optional live-Chrome driver (browser.driver: chrome_use) needs the third-party
  `chrome-use` CLI (Apache-2.0, https://github.com/leeguooooo/chrome-use).

  This script does NOT pipe a remote script into Invoke-Expression. It:
    1. resolves the release tag (latest, or the one you pass),
    2. downloads chrome-use-win32-x64.tar.gz and chrome-use-win32-x64.tar.gz.sha256 from
       https://github.com/leeguooooo/chrome-use/releases (HTTPS, github.com only),
    3. verifies SHA-256 and stops on any mismatch (the archive is deleted),
    4. extracts it with the Windows built-in tar and copies the files into a LOCAL folder
       (default %LOCALAPPDATA%\Programs\chrome-use; no admin rights),
    5. prints the version and what to do next.

  It does NOT: edit PATH (unless -AddToPath), install the browser extension, register the
  native-messaging host, install any "skill", or touch Chrome. Those are separate, deliberate steps.

.PARAMETER Version
  Release tag such as v1.5.157. Default: the latest release.

.PARAMETER InstallDir
  Where to put chrome-use.exe. Default: $env:LOCALAPPDATA\Programs\chrome-use

.PARAMETER AddToPath
  Also append InstallDir to your *user* PATH (off by default; OmniBrain finds the default folder itself).

.PARAMETER DryRun
  Resolve the version and print the URLs and destination, download nothing.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\scripts\install_chrome_use.ps1 -DryRun
  powershell -ExecutionPolicy Bypass -File .\scripts\install_chrome_use.ps1
#>
[CmdletBinding()]
param(
    [string]$Version = 'latest',
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA 'Programs\chrome-use'),
    [switch]$AddToPath,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$Repo = 'leeguooooo/chrome-use'
$Asset = 'chrome-use-win32-x64.tar.gz'

function Fail([string]$Message) {
    Write-Host "ERROR: $Message" -ForegroundColor Red
    exit 1
}

if (-not [Environment]::Is64BitOperatingSystem) { Fail 'This release is Windows x64 only.' }
if (-not (Get-Command tar.exe -ErrorAction SilentlyContinue)) { Fail 'tar.exe was not found (it ships with Windows 10 1803 and later).' }

# 1. which release
if ($Version -eq 'latest') {
    try {
        $release = Invoke-RestMethod -UseBasicParsing -Uri "https://api.github.com/repos/$Repo/releases/latest" -Headers @{ 'User-Agent' = 'omnibrain-install' }
    } catch {
        Fail "Could not read the latest release from GitHub: $($_.Exception.Message)"
    }
    $Version = [string]$release.tag_name
}
if ($Version -notmatch '^v\d+(\.\d+){1,3}([-.][0-9A-Za-z.]+)?$') { Fail "Unexpected release tag '$Version'." }

$base = "https://github.com/$Repo/releases/download/$Version"
$archiveUrl = "$base/$Asset"
$shaUrl = "$base/$Asset.sha256"

Write-Host "chrome-use release : $Version"
Write-Host "archive            : $archiveUrl"
Write-Host "checksum           : $shaUrl"
Write-Host "install to         : $InstallDir"
if ($DryRun) {
    Write-Host 'Dry run: nothing was downloaded or changed.'
    exit 0
}

# 2. download into a private temp folder
$work = Join-Path ([IO.Path]::GetTempPath()) ("chrome-use-install-" + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $work | Out-Null
try {
    $archive = Join-Path $work $Asset
    $shaFile = Join-Path $work "$Asset.sha256"
    try {
        Invoke-WebRequest -UseBasicParsing -Uri $archiveUrl -OutFile $archive
        Invoke-WebRequest -UseBasicParsing -Uri $shaUrl -OutFile $shaFile
    } catch {
        Fail "Download failed: $($_.Exception.Message)"
    }

    # 3. verify
    $line = (Get-Content -LiteralPath $shaFile -Raw).Trim()
    if ($line -notmatch '^([0-9A-Fa-f]{64})\b') { Fail "The .sha256 file is not in the expected format: '$line'" }
    $expected = $Matches[1].ToLowerInvariant()
    $actual = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expected) {
        Remove-Item -LiteralPath $archive -Force
        Fail "SHA-256 mismatch. expected $expected but got $actual. The archive was deleted and nothing was installed."
    }
    Write-Host "SHA-256 verified   : $actual"

    # 4. extract + copy
    $extract = Join-Path $work 'x'
    New-Item -ItemType Directory -Path $extract | Out-Null
    & tar.exe -xzf $archive -C $extract
    if ($LASTEXITCODE -ne 0) { Fail 'tar failed to extract the archive.' }
    $exe = Get-ChildItem -LiteralPath $extract -Recurse -Filter 'chrome-use.exe' | Select-Object -First 1
    if (-not $exe) { Fail 'chrome-use.exe was not found inside the archive.' }

    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    Copy-Item -Path (Join-Path $exe.DirectoryName '*') -Destination $InstallDir -Recurse -Force
    $installed = Join-Path $InstallDir 'chrome-use.exe'
    if (-not (Test-Path -LiteralPath $installed)) { Fail "Copy to $InstallDir failed." }
}
finally {
    Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
}

# 5. optional PATH (user scope, appended, existing entries untouched)
if ($AddToPath) {
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    $parts = @(); if ($userPath) { $parts = $userPath.Split(';') | Where-Object { $_ } }
    if ($parts -notcontains $InstallDir) {
        [Environment]::SetEnvironmentVariable('Path', (($parts + $InstallDir) -join ';'), 'User')
        Write-Host "Added $InstallDir to your user PATH (open a new terminal to see it)."
    }
}

Write-Host ''
& $installed --version
Write-Host ''
Write-Host "Installed: $installed"
Write-Host 'Next (you do these, once):'
Write-Host "  1. & `"$installed`" extension install        # registers the native-messaging host for your user (HKCU)"
Write-Host '  2. In Chrome, add the "chrome-use" extension from the Chrome Web Store:'
Write-Host '     https://chromewebstore.google.com/detail/chrome-use/knfcmbamhjmaonkfnjhldjedeobeafmk'
Write-Host '  3. Check:  python scripts\live_chrome_check.py'
