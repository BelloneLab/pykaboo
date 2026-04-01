[CmdletBinding()]
param(
    [string]$Version = "dev",
    [string]$PythonExe = "python",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptRoot
$specPath = Join-Path $repoRoot "camApp-live-detection.spec"
$distDir = Join-Path $repoRoot "dist"
$buildDir = Join-Path $repoRoot "build"
$releaseDir = Join-Path $repoRoot "release"
$exePath = Join-Path $distDir "CamAppLiveDetection.exe"
$safeVersion = ($Version -replace '[^A-Za-z0-9._-]', '-').Trim('-')
if ([string]::IsNullOrWhiteSpace($safeVersion)) {
    $safeVersion = "dev"
}
$artifactStem = "camApp-live-detection-$safeVersion-windows-x64"
$zipPath = Join-Path $releaseDir "$artifactStem.zip"
$hashPath = Join-Path $releaseDir "$artifactStem.sha256"
$warnSource = Join-Path $buildDir "camApp-live-detection\warn-camApp-live-detection.txt"
$warnTarget = Join-Path $releaseDir "$artifactStem-warn.txt"

if (-not (Test-Path $specPath)) {
    throw "Spec file not found: $specPath"
}

if ($Clean) {
    foreach ($path in @($distDir, $buildDir, $releaseDir)) {
        if (Test-Path $path) {
            Remove-Item -LiteralPath $path -Recurse -Force
        }
    }
}

New-Item -ItemType Directory -Path $releaseDir -Force | Out-Null

$probeScript = @'
import importlib
import importlib.util
import json
import sys

result = {
    "python": sys.executable,
    "version": sys.version.split()[0],
    "pyspin_origin": None,
    "pyspin_search_locations": [],
    "pyspin_usable": False,
    "pyspin_error": "",
}

try:
    spec = importlib.util.find_spec("PySpin")
except Exception as exc:
    spec = None
    result["pyspin_error"] = repr(exc)
else:
    if spec is not None:
        result["pyspin_origin"] = spec.origin
        result["pyspin_search_locations"] = list(getattr(spec, "submodule_search_locations", []) or [])

try:
    module = importlib.import_module("PySpin")
    if hasattr(module, "System"):
        result["pyspin_usable"] = True
    else:
        module = importlib.import_module("PySpin.PySpin")
        result["pyspin_usable"] = hasattr(module, "System")
except Exception as exc:
    if not result["pyspin_error"]:
        result["pyspin_error"] = repr(exc)

print(json.dumps(result))
'@

$probeJson = $probeScript | & $PythonExe -
if ($LASTEXITCODE -ne 0) {
    throw "Python preflight probe failed for $PythonExe"
}

$probe = $probeJson | ConvertFrom-Json
Write-Host "Build interpreter: $($probe.python) [$($probe.version)]"
if ($probe.pyspin_usable) {
    Write-Host "PySpin probe: usable"
} elseif ($probe.pyspin_origin -or ($probe.pyspin_search_locations | Measure-Object).Count -gt 0) {
    Write-Warning ("PySpin is visible to the build interpreter but not usable. " +
        "FLIR Spinnaker cameras will not work in the compiled EXE. " +
        "Origin: {0}. Error: {1}" -f $probe.pyspin_origin, $probe.pyspin_error)
} else {
    Write-Host "PySpin probe: not installed in the build interpreter"
}

$pyInstallerArgs = @("-m", "PyInstaller", "--noconfirm")
if ($Clean) {
    $pyInstallerArgs += "--clean"
}
$pyInstallerArgs += $specPath

Write-Host "Building CamApp Live Detection with $PythonExe"
& $PythonExe @pyInstallerArgs
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller exited with code $LASTEXITCODE"
}

if (-not (Test-Path $exePath)) {
    throw "Build did not produce $exePath"
}

if (Test-Path $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}

Compress-Archive -Path $exePath -DestinationPath $zipPath -Force
$hash = Get-FileHash -LiteralPath $zipPath -Algorithm SHA256
Set-Content -LiteralPath $hashPath -Value ("{0} *{1}" -f $hash.Hash.ToLowerInvariant(), (Split-Path -Leaf $zipPath)) -Encoding ascii

if (Test-Path $warnSource) {
    Copy-Item -LiteralPath $warnSource -Destination $warnTarget -Force
} else {
    Set-Content -LiteralPath $warnTarget -Value "No PyInstaller warnings captured." -Encoding utf8
}

Write-Host "Build complete"
Write-Host "Executable: $exePath"
Write-Host "Release zip: $zipPath"
Write-Host "Checksum: $hashPath"
if (Test-Path $warnTarget) {
    Write-Host "Warnings: $warnTarget"
}
