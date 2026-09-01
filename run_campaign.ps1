<#
    run_campaign.ps1 - unattended driver for the full thesis reproduction.

    Runs every stage sequentially on the one GPU, tees each to logs/<stage>.log,
    and writes logs/<stage>.done on success so a re-run resumes rather than
    restarts. Safe to kill and relaunch; never run two stages concurrently.

    ASCII only on purpose: PowerShell 5.1 reads .ps1 as ANSI unless the file has
    a BOM, so non-ASCII characters here break the parser.

    Usage:
        powershell -ExecutionPolicy Bypass -File run_campaign.ps1
        powershell -ExecutionPolicy Bypass -File run_campaign.ps1 -Only 4a
        powershell -ExecutionPolicy Bypass -File run_campaign.ps1 -List
#>

param(
    [string]$Only = "",
    [switch]$List,
    [switch]$Force
)

$ErrorActionPreference = "Continue"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python   = "C:\Users\admin\anaconda3\envs\fsgrade\python.exe"
$LogDir   = Join-Path $RepoRoot "logs"

$env:FRUITVISION_ROOT = "C:\Users\admin\Desktop\Amr Samir\FruitVision"
$env:PYTHONUNBUFFERED  = "1"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# ------------------------------------------------------------------ #
#  Stage list
# ------------------------------------------------------------------ #
#
# Ordered so the two things needed for writing land first: a complete
# citable result set (4a), then the executed notebook (5). The remaining
# seeds, the aggregate and the variants follow.
#
# 'kind' is 'script' (reproduce_thesis.py) or 'notebook' (nbconvert).

$Stages = @(
    @{ id="4a"; kind="script";   desc="Seed 42, full (backbone ablation + species CV + single-species)";
       args=@("--seed","42") }

    @{ id="5";  kind="notebook"; desc="Execute notebooks/main_experiment.ipynb top to bottom";
       nb="notebooks\main_experiment.ipynb" }

    @{ id="4b"; kind="script";   desc="Seed 1337, full";
       args=@("--seed","1337") }

    @{ id="4c"; kind="script";   desc="Seed 2024, full";
       args=@("--seed","2024") }

    @{ id="4d"; kind="script";   desc="Aggregate the three seeds into results/seed_aggregate.json";
       args=@("--aggregate") }

    @{ id="4e"; kind="script";   desc="Variant: --freeze-bn-stats";
       args=@("--seed","42","--freeze-bn-stats","--skip-expensive") }

    @{ id="4f"; kind="script";   desc="Variant: LOSO validation protocol";
       args=@("--seed","42","--val-protocol","loso","--skip-expensive") }

    @{ id="4g"; kind="script";   desc="Variant: LOSO + frozen BN stats";
       args=@("--seed","42","--val-protocol","loso","--freeze-bn-stats","--skip-expensive") }

    @{ id="6";  kind="notebook"; desc="Render notebooks/thesis_results.ipynb from finished runs";
       nb="notebooks\thesis_results.ipynb"; optional=$true }
)

if ($List) {
    ""
    "  Stage   Status     Description"
    "  -----   -------    -----------"
    foreach ($s in $Stages) {
        $done = Test-Path (Join-Path $LogDir "$($s.id).done")
        $mark = if ($done) { "DONE   " } else { "pending" }
        "  {0,-6}  {1}    {2}" -f $s.id, $mark, $s.desc
    }
    ""
    exit 0
}

# ------------------------------------------------------------------ #
#  Keep the machine awake - this campaign runs for days
# ------------------------------------------------------------------ #
try {
    powercfg /change standby-timeout-ac 0   | Out-Null
    powercfg /change hibernate-timeout-ac 0 | Out-Null
    powercfg /change monitor-timeout-ac 0   | Out-Null
    Write-Host "Sleep/hibernate disabled on AC power."
} catch {
    Write-Warning "Could not change power settings: $_"
}

# ------------------------------------------------------------------ #
#  Runner
# ------------------------------------------------------------------ #

function Invoke-Stage($stage) {
    $id       = $stage.id
    $logFile  = Join-Path $LogDir "$id.log"
    $doneFile = Join-Path $LogDir "$id.done"

    if ((Test-Path $doneFile) -and -not $Force) {
        Write-Host "[$id] already complete - skipping. (-Force to re-run)"
        return $true
    }

    if ($stage.kind -eq "notebook") {
        $nbPath = Join-Path $RepoRoot $stage.nb
        if (-not (Test-Path $nbPath)) {
            if ($stage.optional) {
                Write-Host "[$id] notebook not present yet - skipping (optional)."
                return $true
            }
            Write-Warning "[$id] notebook missing: $nbPath"
            return $false
        }
        $exe  = $Python
        $argv = @("-m","jupyter","nbconvert","--to","notebook","--execute","--inplace",
                  "--ExecutePreprocessor.kernel_name=fsgrade",
                  "--ExecutePreprocessor.timeout=-1",
                  $nbPath)
    }
    else {
        $exe  = $Python
        $argv = @((Join-Path $RepoRoot "scripts\reproduce_thesis.py")) + $stage.args
    }

    $started = Get-Date
    Write-Host ""
    Write-Host ("=" * 72)
    Write-Host "[$id] $($stage.desc)"
    Write-Host "[$id] started $started"
    Write-Host "[$id] log -> $logFile"
    Write-Host ("=" * 72)

    "=== stage $id | $($stage.desc) | started $started ===" |
        Out-File -FilePath $logFile -Encoding utf8

    # Redirect through cmd so stderr interleaves into the log without
    # PowerShell 5.1 wrapping each native stderr line in an ErrorRecord.
    $quoted = ($argv | ForEach-Object { '"' + $_ + '"' }) -join " "
    & cmd /c "`"$exe`" $quoted >> `"$logFile`" 2>&1"
    $code = $LASTEXITCODE

    $elapsed = (Get-Date) - $started
    $summary = "[$id] exit=$code after " + $elapsed.ToString("hh\:mm\:ss")
    Write-Host $summary
    $summary | Out-File -FilePath $logFile -Encoding utf8 -Append

    if ($code -eq 0) {
        "completed $(Get-Date) after $elapsed" | Out-File -FilePath $doneFile -Encoding utf8
        return $true
    }
    Write-Warning "[$id] FAILED with exit $code - see $logFile"
    return $false
}

$toRun = if ($Only) { $Stages | Where-Object { $_.id -eq $Only } } else { $Stages }

if (-not $toRun) {
    Write-Warning "No stage matched '$Only'. Use -List to see stage ids."
    exit 1
}

$campaignStart = Get-Date
foreach ($stage in $toRun) {
    if (-not (Invoke-Stage $stage)) {
        Write-Warning "Campaign halted at stage $($stage.id). Fix, then re-run to resume."
        exit 1
    }
}

Write-Host ""
Write-Host ("=" * 72)
Write-Host "CAMPAIGN COMPLETE in $((Get-Date) - $campaignStart)"
Write-Host ("=" * 72)
exit 0
