param(
    [string] $Version,
    [switch] $OneFile,
    [switch] $Clean,
    [switch] $SkipInstall,
    [switch] $RecreateVenv,
    [switch] $Lite
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Resolve-BuildVersion {
    param([string] $RequestedVersion)

    if ($RequestedVersion) { return $RequestedVersion }

    $pyprojectPath = Join-Path $RepoRoot "pyproject.toml"
    if (Test-Path -LiteralPath $pyprojectPath) {
        $pyprojectText = Get-Content -LiteralPath $pyprojectPath -Raw
        $versionMatch = [regex]::Match($pyprojectText, '(?m)^\s*version\s*=\s*"([^"]+)"')
        if ($versionMatch.Success -and $versionMatch.Groups[1].Value.Trim()) {
            return $versionMatch.Groups[1].Value.Trim()
        }
    }

    $gitVersion = (& git @('describe', '--tags', '--always') 2>$null)
    if ($LASTEXITCODE -eq 0 -and $gitVersion) {
        $trimmed = ($gitVersion | Select-Object -First 1).Trim()
        if ($trimmed) { return $trimmed }
    }

    throw "ABORT(A14): version is required. Pass -Version, add project.version to pyproject.toml, or ensure git describe works."
}

function Get-SafeFileNamePart {
    param([string] $Value)
    return ($Value -replace '[^0-9A-Za-z._-]', '-')
}

function Get-FileSizeText {
    param([long] $Bytes)
    $mb = [math]::Round($Bytes / 1MB, 2)
    return "$Bytes bytes ($mb MB)"
}

function New-BuildVirtualEnvironment {
    param([string] $TargetPath)

    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        & py -3.11 -m venv $TargetPath
        return $LASTEXITCODE
    }

    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        & python -m venv $TargetPath
        return $LASTEXITCODE
    }

    throw "ABORT(A13): neither py launcher nor python command was found for build venv creation"
}

$RepoRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
Set-Location -LiteralPath $RepoRoot

if ($PSVersionTable.PSEdition -ne "Core" -or $PSVersionTable.PSVersion.Major -ne 7) {
    throw "ABORT(A0): build script requires PowerShell 7 Core. Run with pwsh."
}

$BuildVenv = Join-Path $RepoRoot ".venv_build"
if ($RecreateVenv -and (Test-Path -LiteralPath $BuildVenv)) {
    $resolvedBuildVenv = Resolve-Path -LiteralPath $BuildVenv
    if (-not $resolvedBuildVenv.Path.StartsWith($RepoRoot.Path, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "ABORT(A11): refusing to remove build venv outside repo: $resolvedBuildVenv"
    }
    Write-Host "Removing build virtual environment: $BuildVenv"
    Remove-Item -LiteralPath $BuildVenv -Recurse -Force
}
$Python = Join-Path $BuildVenv "Scripts\python.exe"
$AppEntry = Join-Path $RepoRoot "run_windows_remote_app.py"
$BaseAppName = if ($Lite) { "Deep-Live-Cam-Remote-Lite" } else { "Deep-Live-Cam-Remote" }
$IconPath = Join-Path $RepoRoot "windows_app\icon.ico"
$ThemePath = Join-Path $RepoRoot "windows_app\dark_theme.qss"

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Host "Creating build virtual environment: $BuildVenv"
    $venvExitCode = New-BuildVirtualEnvironment -TargetPath $BuildVenv
    if ($venvExitCode -ne 0) { throw "ABORT(A13): failed to create build venv ($venvExitCode)" }
}

if (-not $SkipInstall) {
    Write-Host "Installing build requirements from requirements-build.txt"
    & $Python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "ABORT(A13): pip upgrade failed ($LASTEXITCODE)" }
    & $Python -m pip install -r (Join-Path $RepoRoot "requirements-build.txt")
    if ($LASTEXITCODE -ne 0) { throw "ABORT(A13): build requirements install failed ($LASTEXITCODE)" }
}

$resolvedVersion = Resolve-BuildVersion -RequestedVersion $Version
$safeVersion = Get-SafeFileNamePart -Value $resolvedVersion
$pyVersionShort = (& $Python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if ($LASTEXITCODE -ne 0 -or -not $pyVersionShort) { throw "ABORT(A13): failed to resolve build Python version" }
$pyVersionShort = ($pyVersionShort | Select-Object -First 1).Trim()
$ArtifactName = "$BaseAppName-$safeVersion-py$pyVersionShort"
$DistRoot = Join-Path $RepoRoot "dist"
$DistPath = Join-Path $DistRoot $safeVersion
$WorkPath = Join-Path $RepoRoot "build"
$SpecPath = Join-Path $WorkPath "specs"

if (-not (Test-Path -LiteralPath $DistPath)) {
    New-Item -ItemType Directory -Path $DistPath | Out-Null
}
if (-not (Test-Path -LiteralPath $SpecPath)) {
    New-Item -ItemType Directory -Path $SpecPath | Out-Null
}

$modeArg = if ($OneFile) { "--onefile" } else { "--onedir" }
$cleanArgs = if ($Clean) { @("--clean") } else { @() }
$separator = ";"

$args = @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--optimize=2",
    "--noupx",
    $modeArg,
    "--windowed",
    "--name", $ArtifactName,
    "--distpath", $DistPath,
    "--workpath", $WorkPath,
    "--specpath", $SpecPath,
    "--add-data", "$ThemePath${separator}windows_app",
    "--add-data", "$IconPath${separator}windows_app",
    "--hidden-import", "PySide6.QtMultimedia",
    "--hidden-import", "PySide6.QtMultimediaWidgets",
    "--hidden-import", "websockets"
)

if ($Lite) {
    $args += @(
        "--exclude-module", "cv2",
        "--exclude-module", "numpy",
        "--exclude-module", "pyvirtualcam"
    )
} else {
    $args += @(
        "--hidden-import", "cv2",
        "--hidden-import", "numpy",
        "--hidden-import", "pyvirtualcam"
    )
}

if (Test-Path -LiteralPath $IconPath) {
    $args += @("--icon", $IconPath)
}

$args += $cleanArgs
$args += $AppEntry

if ($Lite) { Write-Host "Lite build: live webcam dependencies are excluded (cv2, numpy, pyvirtualcam)." }
Write-Host "Running PyInstaller ($modeArg) for $ArtifactName"
Write-Host "Version: $resolvedVersion"
Write-Host "Python: $pyVersionShort"
Write-Host "Dist path: $DistPath"
& $Python @args
if ($LASTEXITCODE -ne 0) { throw "ABORT(A13): PyInstaller failed ($LASTEXITCODE)" }

if ($OneFile) {
    $primaryOutput = Join-Path $DistPath "$ArtifactName.exe"
} else {
    $primaryOutput = Join-Path (Join-Path $DistPath $ArtifactName) "$ArtifactName.exe"
}

if (-not (Test-Path -LiteralPath $primaryOutput -PathType Leaf)) {
    throw "ABORT(A14): expected built executable not found: $primaryOutput"
}

$builtItem = Get-Item -LiteralPath $primaryOutput
Write-Host "== Build complete =="
Write-Host "Primary output: $($builtItem.FullName)"
Write-Host "Size: $(Get-FileSizeText -Bytes $builtItem.Length)"
Write-Host "Mode: $(if ($OneFile) { 'onefile' } else { 'onedir' })"
Write-Host "Flavor: $(if ($Lite) { 'Lite' } else { 'Full' })"
Write-Host "Version: $resolvedVersion"
