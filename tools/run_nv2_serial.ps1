param(
  [string[]]$Datasets = @("2wikimultihopqa", "hotpotqa", "musique", "nq_rear", "popqa"),
  [string]$EmbeddingName = "nvidia/NV-Embed-v2",
  [string]$LLMName = "meta/llama-3.3-70b-instruct",
  [string]$LLMBaseUrl = "",
  [string]$SaveRoot = "outputs",
  [int]$OfflineLLMWorkers = 16,
  [int]$OnlineQAWorkers = 8,
  [switch]$RetrievalOnly,
  [switch]$SkipQA,
  [switch]$ForceRebuild,
  [switch]$CleanOutputs,
  [switch]$Resume
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

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
  if (-not [string]::IsNullOrWhiteSpace($env:CONDA_PREFIX)) {
    $candidate = Join-Path $env:CONDA_PREFIX "python.exe"
    if (Test-Path $candidate) { $pythonExe = $candidate }
  }
  if ($null -eq $pythonExe) {
    $pythonExe = (Get-Command python -ErrorAction Stop).Source
  }

  Write-Host ("[RUN] {0} | {1} | tag={2} | force_rebuild={3}" -f $dataset, $runMode, $runTag, $forceIndexFromScratch)
  Write-Host ("      stdout={0}" -f $stdout)
  Write-Host ("      stderr={0}" -f $stderr)
  Write-Host ("      python_exe={0}" -f $pythonExe)
  Write-Host ("      cmd={0} {1}" -f $pythonExe, ($args -join " "))

  $p = Start-Process -FilePath $pythonExe -ArgumentList $args -NoNewWindow -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
  $p.WaitForExit()
  $exitCode = $p.ExitCode
  if ($null -eq $exitCode) { $exitCode = "unknown" }
  if ($exitCode -ne 0 -and $exitCode -ne "unknown") {
    Write-Host ("[FAIL] exit={0} (showing last 120 lines of stderr)" -f $exitCode)
    if (Test-Path $stderr) { Get-Content -Tail 120 $stderr | ForEach-Object { Write-Host $_ } }
    throw ("Run failed: dataset={0} mode={1} tag={2} (exit={3}). See: {4} / {5}" -f $dataset, $runMode, $runTag, $exitCode, $stdout, $stderr)
  }
  if ($exitCode -eq "unknown") {
    Write-Host "[WARN] Process ExitCode is unknown. Check logs for details."
    if (Test-Path $stderr) { Get-Content -Tail 120 $stderr | ForEach-Object { Write-Host $_ } }
    throw ("Run failed: dataset={0} mode={1} tag={2} (exit=unknown). See: {3} / {4}" -f $dataset, $runMode, $runTag, $stdout, $stderr)
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

    Run-One -dataset $ds -runMode "retrieval_only" -runTag $retrTag -forceIndexFromScratch:$ForceRebuild

    if (-not $RetrievalOnly -and -not $SkipQA) {
      Run-One -dataset $ds -runMode "rag_qa" -runTag $qaTag -forceIndexFromScratch:$false
    }
  }

  Write-Host "[DONE] NV2 serial runs finished."
} finally {
  Pop-Location
}
