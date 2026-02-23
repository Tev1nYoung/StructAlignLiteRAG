param(
  [string[]]$Datasets = @("sample", "case_study_university", "2wikimultihopqa", "hotpotqa", "musique", "nq_rear", "popqa"),
  [string]$EmbeddingName = "nvidia/NV-Embed-v2",
  [string]$LLMName = "meta/llama-3.3-70b-instruct",
  [string]$LLMBaseUrl = "",
  [string]$SaveRoot = "outputs",
  [int]$OfflineLLMWorkers = 16,
  [int]$OnlineQAWorkers = 8,
  [string]$CondaEnvName = "",
  [switch]$RetrievalOnly,
  [switch]$SkipQA,
  [switch]$ForceRebuild,
  [switch]$CleanOutputs,
  [switch]$Resume
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-CondaExe {
  # Prefer conda.exe (native) rather than PowerShell function/module wrapper.
  if (-not [string]::IsNullOrWhiteSpace($env:CONDA_EXE) -and (Test-Path $env:CONDA_EXE)) {
    return $env:CONDA_EXE
  }
  if (-not [string]::IsNullOrWhiteSpace($env:CONDA_PREFIX)) {
    $cand = Join-Path $env:CONDA_PREFIX "Scripts\\conda.exe"
    if (Test-Path $cand) { return $cand }
    $cand = Join-Path $env:CONDA_PREFIX "condabin\\conda.bat"
    if (Test-Path $cand) { return $cand }
  }

  $cmd = Get-Command conda.exe -ErrorAction SilentlyContinue
  if ($null -ne $cmd -and -not [string]::IsNullOrWhiteSpace($cmd.Source) -and (Test-Path $cmd.Source)) {
    return $cmd.Source
  }
  $cmd = Get-Command conda.bat -ErrorAction SilentlyContinue
  if ($null -ne $cmd -and -not [string]::IsNullOrWhiteSpace($cmd.Source) -and (Test-Path $cmd.Source)) {
    return $cmd.Source
  }

  throw "Cannot locate conda executable (conda.exe/conda.bat)."
}

function Sanitize-ModelName([string]$name) {
  if ([string]::IsNullOrWhiteSpace($name)) { return "none" }
  $s = $name.Trim().Replace("\", "_").Replace("/", "_")
  $s = [Regex]::Replace($s, "[^A-Za-z0-9._-]+", "_")
  $s = [Regex]::Replace($s, "_+", "_").Trim("_")
  if ([string]::IsNullOrWhiteSpace($s)) { return "none" }
  return $s
}

function Run-One([string]$dataset, [string]$runMode, [string]$runTag, [bool]$forceIndexFromScratch) {
  $llmSan = Sanitize-ModelName $LLMName
  $embSan = Sanitize-ModelName $EmbeddingName

  $predFile = if ($runMode -eq "retrieval_only") { "retrieval_predictions.json" } else { "qa_predictions.json" }
  $predPath = Join-Path $SaveRoot (Join-Path $dataset (Join-Path ("{0}_{1}" -f $llmSan, $embSan) (Join-Path "metrics\\runs" (Join-Path $runTag $predFile))))

  if ($Resume -and (Test-Path $predPath)) {
    Write-Host ("[SKIP] {0} | {1} | {2} exists: {3}" -f $dataset, $runMode, $runTag, $predPath)
    return
  }

  $logRoot = Join-Path "logs-nv2" $dataset
  New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
  $ts = Get-Date -Format "yyyyMMdd_HHmmss"
  $stdout = Join-Path $logRoot ("{0}_{1}_{2}.out.log" -f $ts, $runMode, $runTag)
  $stderr = Join-Path $logRoot ("{0}_{1}_{2}.err.log" -f $ts, $runMode, $runTag)

  $args = @(
    "main.py",
    "--dataset", $dataset,
    "--run_mode", $runMode,
    "--embedding_name", $EmbeddingName,
    "--llm_name", $LLMName,
    "--save_root", $SaveRoot,
    "--run_tag", $runTag,
    "--force_index_from_scratch", ($(if ($forceIndexFromScratch) { "true" } else { "false" })),
    "--offline_llm_workers", $OfflineLLMWorkers,
    "--online_qa_workers", $OnlineQAWorkers
  )

  if (-not [string]::IsNullOrWhiteSpace($LLMBaseUrl)) {
    $args += @("--llm_base_url", $LLMBaseUrl)
  }

  $pythonExe = $null
  $useCondaRun = $false
  $envName = $CondaEnvName
  if ([string]::IsNullOrWhiteSpace($envName)) {
    $envName = $env:CONDA_DEFAULT_ENV
  }
  if ([string]::IsNullOrWhiteSpace($envName)) {
    $envName = "hipporag"
  }

  $condaPrefix = $env:CONDA_PREFIX
  $candidates = @()

  # Prefer the env python if CONDA_PREFIX points to base (common when calling a fresh powershell process).
  if (-not [string]::IsNullOrWhiteSpace($condaPrefix) -and $envName -ne "base") {
    $candidates += (Join-Path $condaPrefix (Join-Path "envs\\$envName" "python.exe"))
    $condaRoot = Split-Path -Parent $condaPrefix
    if (-not [string]::IsNullOrWhiteSpace($condaRoot)) {
      $candidates += (Join-Path $condaRoot (Join-Path "envs\\$envName" "python.exe"))
    }
  }
  if (-not [string]::IsNullOrWhiteSpace($condaPrefix)) {
    $candidates += (Join-Path $condaPrefix "python.exe")
  }
  $candidates += @(
    (Get-Command python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue)
  )

  foreach ($cand in $candidates) {
    if (-not [string]::IsNullOrWhiteSpace($cand) -and (Test-Path $cand)) {
      $pythonExe = $cand
      break
    }
  }
  # If we can't reliably locate env python, fall back to `conda run -n <env> python ...`.
  # This is robust when the script is launched from a fresh PowerShell process and CONDA env vars don't reflect the prompt.
  if ($envName -ne "base") {
    if ($null -eq $pythonExe) {
      $useCondaRun = $true
    } else {
      $envNeedle = ("\\envs\\{0}\\" -f $envName).ToLowerInvariant()
      if ($pythonExe.ToLowerInvariant().IndexOf($envNeedle) -lt 0) {
        $useCondaRun = $true
      }
    }
  }
  if (-not $useCondaRun -and ($null -eq $pythonExe)) {
    throw "Cannot locate python.exe. Please run: conda activate hipporag, then run this script again."
  }

  Write-Host ("[RUN] {0} | {1} | tag={2} | force_rebuild={3}" -f $dataset, $runMode, $runTag, $forceIndexFromScratch)
  Write-Host ("      stdout={0}" -f $stdout)
  Write-Host ("      stderr={0}" -f $stderr)
  Write-Host ("      conda_env={0} conda_prefix={1}" -f $envName, $condaPrefix)
  if ($useCondaRun) {
    $condaExe = Resolve-CondaExe
    # Use python -u to force flush (helps real-time log writing).
    $condaArgList = @("run", "--no-capture-output", "-n", $envName, "python", "-u") + $args
    Write-Host ("      mode=conda_run conda_exe={0}" -f $condaExe)
    Write-Host ("      cmd={0} {1}" -f $condaExe, ($condaArgList -join " "))
    $p = Start-Process -FilePath $condaExe -ArgumentList $condaArgList -NoNewWindow -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $p.WaitForExit()
    $exitCode = $p.ExitCode
  } else {
    Write-Host ("      mode=python_exe python_exe={0}" -f $pythonExe)
    Write-Host ("      cmd={0} -u {1}" -f $pythonExe, ($args -join " "))
    $p = Start-Process -FilePath $pythonExe -ArgumentList @("-u") + $args -NoNewWindow -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $p.WaitForExit()
    $exitCode = $p.ExitCode
  }

  if ($null -eq $exitCode) { $exitCode = 0 }
  if ($exitCode -ne 0) {
    Write-Host ("[FAIL] exit={0} (showing last 120 lines of stderr)" -f $exitCode)
    if (Test-Path $stderr) { Get-Content -Tail 120 $stderr | ForEach-Object { Write-Host $_ } }
    throw ("Run failed: dataset={0} mode={1} tag={2} (exit={3}). See: {4} / {5}" -f $dataset, $runMode, $runTag, $exitCode, $stdout, $stderr)
  }

  if (-not (Test-Path $predPath)) {
    Write-Host "[FAIL] run finished but predictions file not found (showing last 80 lines of stderr)"
    if (Test-Path $stderr) { Get-Content -Tail 80 $stderr | ForEach-Object { Write-Host $_ } }
    throw ("Run finished but predictions missing: {0}" -f $predPath)
  }
}

function Remove-IfExists([string]$path) {
  if (Test-Path $path) {
    Remove-Item -Recurse -Force $path
  }
}

if ($RetrievalOnly -and $SkipQA) {
  throw "Invalid flags: -RetrievalOnly and -SkipQA cannot both be set."
}

$repoRoot = Split-Path -Parent $PSScriptRoot  # tools/ -> repo root
Push-Location $repoRoot
try {
  foreach ($dsRaw in $Datasets) {
    $dsVal = if ($null -eq $dsRaw) { "" } else { [string]$dsRaw }
    $ds = $dsVal.Trim()
    if ([string]::IsNullOrWhiteSpace($ds)) { continue }

    if ($CleanOutputs) {
      $llmSan = Sanitize-ModelName $LLMName
      $embSan = Sanitize-ModelName $EmbeddingName
      $outDir = Join-Path $SaveRoot (Join-Path $ds ("{0}_{1}" -f $llmSan, $embSan))
      Write-Host ("[CLEAN] removing outputs: {0}" -f $outDir)
      Remove-IfExists $outDir
    }

    $retrTag = "nv2_retrieval_{0}" -f $ds
    $qaTag = "nv2_rag_qa_{0}" -f $ds

    if ($RetrievalOnly -or $SkipQA) {
      Run-One -dataset $ds -runMode "retrieval_only" -runTag $retrTag -forceIndexFromScratch:$ForceRebuild
    } else {
      # Default: full end-to-end
      Run-One -dataset $ds -runMode "rag_qa" -runTag $qaTag -forceIndexFromScratch:$ForceRebuild
    }
  }

  Write-Host "[DONE] NV2 serial runs finished."
} finally {
  Pop-Location
}
