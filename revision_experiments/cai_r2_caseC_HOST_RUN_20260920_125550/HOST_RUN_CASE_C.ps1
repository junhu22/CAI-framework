$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# AUTHOR CONFIRMATION REQUIRED.
# Replace only this value with a script block implementing the exact ANSYS
# invocation already validated in the real Windows desktop session.
# Contract: param($InputPath, $OutputLogPath, $JobName, $WorkingDirectory)
# Return either an integer exit code or an object with ExitCode and Invocation.
$ANSYS_COMMAND = {
    param($InputPath, $OutputLogPath, $JobName, $WorkingDirectory)

    Push-Location $WorkingDirectory
    try {
        & "D:\Program Files\ANSYS Inc\v160\ansys\bin\winx64\ANSYS160.exe" `
            -b `
            -i $InputPath `
            -o $OutputLogPath `
            -j $JobName `
            -np 1

        $Code = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }

    [pscustomobject]@{
        ExitCode   = $Code
        Invocation = "ANSYS160.exe -b -i <run.mac> -o <ans.log> -j $JobName -np 1"
    }
}

$PackageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$InputsRoot = Join-Path $PackageRoot "inputs"
$OutputsRoot = Join-Path $PackageRoot "outputs"
$ManifestPath = Join-Path $PackageRoot "HOST_RUN_MANIFEST.json"
$ThetaPath = Join-Path $InputsRoot "CASE_C_SELECTED_THETA.csv"
$IndicesPath = Join-Path $InputsRoot "CASE_C_SELECTED_INDICES.json"
$TopCsvPath = Join-Path $OutputsRoot "HOST_CASE_C_FE.csv"
$TopStatusPath = Join-Path $OutputsRoot "HOST_CASE_C_RUN_STATUS.json"

$ExpectedThetaHash = "054bbbcbe4c75a604670faa5d6a63f9e8ec309c0a1dfdd853a968cfc2b427bf5"
$ExpectedIndicesHash = "4188aca464376a1a4679206653f0281c686cbdff7c5e02d84637aca81a10817f"

function Get-Sha256([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $null }
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Write-JsonNoBom([string]$Path, $Value) {
    $Text = $Value | ConvertTo-Json -Depth 12
    [System.IO.File]::WriteAllText($Path, $Text + "`r`n", [System.Text.UTF8Encoding]::new($false))
}

function Test-FinitePositive([double]$Value) {
    return (-not [double]::IsNaN($Value)) -and (-not [double]::IsInfinity($Value)) -and ($Value -gt 0.0)
}

function Get-ModalValues([string]$Path) {
    $Values = New-Object System.Collections.Generic.List[double]
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return @() }
    $Tokens = ([System.IO.File]::ReadAllText($Path) -split "\s+") | Where-Object { $_ -ne "" }
    foreach ($Token in $Tokens) {
        $Parsed = 0.0
        $Ok = [double]::TryParse(
            $Token,
            [System.Globalization.NumberStyles]::Float,
            [System.Globalization.CultureInfo]::InvariantCulture,
            [ref]$Parsed
        )
        if (-not $Ok) { return @() }
        $Values.Add($Parsed)
    }
    return @($Values)
}

function New-FlatRecord($ThetaRow, [string]$CaseName, [string]$JobName) {
    $Record = [ordered]@{
        selection_order = [string]$ThetaRow.selection_order
        chain_position = [string]$ThetaRow.chain_position
        chain_seed = [string]$ThetaRow.chain_seed
        retained_index_zero_based = [string]$ThetaRow.retained_index_zero_based
        e1 = [string]$ThetaRow.e1
        e2 = [string]$ThetaRow.e2
        e3 = [string]$ThetaRow.e3
        case_name = $CaseName
        job_name = $JobName
        process_exit_code = ""
        runtime_s = ""
        solver_status = ""
        run_completed_marker = ""
        zero_error_messages_marker = ""
        license_error_detected = ""
        modal_frequency_count = "0"
        input_run_mac_sha256 = ""
        output_modal_freqs_sha256 = ""
        ans_log_sha256 = ""
        invocation = ""
    }
    for ($Mode = 1; $Mode -le 20; $Mode++) {
        $Record[("fe_f{0:D2}_hz" -f $Mode)] = ""
    }
    return $Record
}

function Write-CumulativeOutputs($Records, [bool]$RunComplete) {
    $Ordered = @($Records | Sort-Object { [int]$_.selection_order })
    if ($Ordered.Count -gt 0) {
        $Ordered | ForEach-Object { [pscustomobject]$_ } | Export-Csv -LiteralPath $TopCsvPath -NoTypeInformation -Encoding UTF8
    }
    $ValidCount = @($Ordered | Where-Object { $_.solver_status -eq "SUCCESS" }).Count
    $FailedCount = $Ordered.Count - $ValidCount
    $Gate = if ($RunComplete -and $Ordered.Count -eq 50 -and $ValidCount -eq 50) { "PASS" } elseif ($RunComplete) { "FAIL" } else { "INCOMPLETE" }
    $Status = [ordered]@{
        schema = "cai-r2-case-c-host-run-status-v1"
        updated_at = [DateTimeOffset]::Now.ToString("o")
        CASE_C_HOST_RUN = $Gate
        run_complete = $RunComplete
        locked_points = 50
        records_written = $Ordered.Count
        valid_20_mode_outputs = $ValidCount
        failed_or_indeterminate = $FailedCount
        retries = 0
        replacements = 0
        extra_points = 0
        metrics_allowed = ($Gate -eq "PASS")
        scientific_protocol = "UNCHANGED"
        launch_configuration = "author-confirmed ANSYS_COMMAND script block"
        result_csv = "outputs/HOST_CASE_C_FE.csv"
        records = @($Ordered | ForEach-Object { [pscustomobject]$_ })
    }
    Write-JsonNoBom $TopStatusPath $Status
}

if ($null -eq $ANSYS_COMMAND -or -not ($ANSYS_COMMAND -is [scriptblock])) {
    throw "AUTHOR ACTION REQUIRED: configure the single `$ANSYS_COMMAND script block in HOST_RUN_CASE_C.ps1 using the exact successful desktop invocation. No ANSYS process was started."
}

if ((Get-Sha256 $ThetaPath) -ne $ExpectedThetaHash) { throw "Locked theta hash mismatch" }
if ((Get-Sha256 $IndicesPath) -ne $ExpectedIndicesHash) { throw "Locked indices hash mismatch" }

$Manifest = Get-Content -Raw -LiteralPath $ManifestPath | ConvertFrom-Json
$ThetaRows = @(Import-Csv -LiteralPath $ThetaPath)
if ($ThetaRows.Count -ne 50) { throw "Expected exactly 50 locked theta rows" }
for ($Index = 0; $Index -lt 50; $Index++) {
    if ([int]$ThetaRows[$Index].selection_order -ne $Index) { throw "Locked theta order mismatch at row $Index" }
}
if ($Manifest.points.Count -ne 50) { throw "Manifest does not contain exactly 50 points" }

New-Item -ItemType Directory -Path $OutputsRoot -Force | Out-Null
$Records = New-Object System.Collections.Generic.List[object]

for ($Index = 0; $Index -lt 50; $Index++) {
    $Theta = $ThetaRows[$Index]
    $Point = $Manifest.points[$Index]
    if ([int]$Point.selection_order -ne $Index) { throw "Manifest point order mismatch at $Index" }

    $CaseName = "case_{0:D3}" -f $Index
    $JobName = "caiC_h{0:D3}" -f $Index
    $InputRunMac = Join-Path $PackageRoot ([string]$Point.run_mac_path)
    $ExpectedRunMacHash = [string]$Point.run_mac_sha256
    if ((Get-Sha256 $InputRunMac) -ne $ExpectedRunMacHash) { throw "Prepared APDL hash mismatch for $CaseName" }

    $WorkDir = Join-Path $OutputsRoot $CaseName
    $AttemptStartPath = Join-Path $WorkDir "HOST_ATTEMPT_STARTED.json"
    $AttemptResultPath = Join-Path $WorkDir "HOST_ATTEMPT_RESULT.json"
    $WorkingRunMac = Join-Path $WorkDir "run.mac"
    $AnsysLog = Join-Path $WorkDir "ans.log"
    $ModalFile = Join-Path $WorkDir "modal_freqs.txt"

    if (Test-Path -LiteralPath $AttemptStartPath) {
        if (Test-Path -LiteralPath $AttemptResultPath) {
            $Prior = Get-Content -Raw -LiteralPath $AttemptResultPath | ConvertFrom-Json
            $Recovered = New-FlatRecord $Theta $CaseName $JobName
            foreach ($Property in $Prior.PSObject.Properties) {
                if ($Recovered.Contains($Property.Name)) { $Recovered[$Property.Name] = [string]$Property.Value }
            }
            $Records.Add($Recovered)
        } else {
            $Indeterminate = New-FlatRecord $Theta $CaseName $JobName
            $Indeterminate.solver_status = "INDETERMINATE_PRIOR_ATTEMPT_NO_RESULT_NO_RETRY"
            $Indeterminate.input_run_mac_sha256 = Get-Sha256 $WorkingRunMac
            $Indeterminate.output_modal_freqs_sha256 = Get-Sha256 $ModalFile
            $Indeterminate.ans_log_sha256 = Get-Sha256 $AnsysLog
            $Records.Add($Indeterminate)
        }
        Write-CumulativeOutputs $Records $false
        continue
    }

    if (Test-Path -LiteralPath $WorkDir) {
        $Existing = @(Get-ChildItem -LiteralPath $WorkDir -Force)
        if ($Existing.Count -gt 0) { throw "Unmarked nonempty point directory blocks safe first attempt: $WorkDir" }
    } else {
        New-Item -ItemType Directory -Path $WorkDir | Out-Null
    }

    Copy-Item -LiteralPath $InputRunMac -Destination $WorkingRunMac
    if ((Get-Sha256 $WorkingRunMac) -ne $ExpectedRunMacHash) { throw "Working APDL hash mismatch for $CaseName" }

    $ThetaRecord = [ordered]@{
        selection_order = $Index
        chain_position = [int]$Theta.chain_position
        chain_seed = [int]$Theta.chain_seed
        retained_index_zero_based = [int]$Theta.retained_index_zero_based
        theta = @([double]$Theta.e1, [double]$Theta.e2, [double]$Theta.e3)
        source_theta_sha256 = $ExpectedThetaHash
        run_mac_sha256 = $ExpectedRunMacHash
    }
    Write-JsonNoBom (Join-Path $WorkDir "theta.json") $ThetaRecord

    $StartedAt = [DateTimeOffset]::Now
    $AttemptStart = [ordered]@{
        schema = "cai-r2-case-c-host-attempt-start-v1"
        started_at = $StartedAt.ToString("o")
        selection_order = $Index
        case_name = $CaseName
        job_name = $JobName
        working_directory = $WorkDir
        input_path = $WorkingRunMac
        output_log_path = $AnsysLog
        run_mac_sha256 = $ExpectedRunMacHash
        no_automatic_retry = $true
    }
    Write-JsonNoBom $AttemptStartPath $AttemptStart

    $Record = New-FlatRecord $Theta $CaseName $JobName
    $Record.input_run_mac_sha256 = $ExpectedRunMacHash
    $Timer = [System.Diagnostics.Stopwatch]::StartNew()
    $CommandExitCode = $null
    $InvocationDescription = ""
    $LaunchException = $null
    try {
        $Returned = @(& $ANSYS_COMMAND -InputPath $WorkingRunMac -OutputLogPath $AnsysLog -JobName $JobName -WorkingDirectory $WorkDir)
        if ($Returned.Count -eq 0) { throw "ANSYS_COMMAND returned no exit code" }
        $CommandResult = $Returned[-1]
        if ($CommandResult -is [int]) {
            $CommandExitCode = [int]$CommandResult
        } elseif ($null -ne $CommandResult.PSObject.Properties["ExitCode"]) {
            $CommandExitCode = [int]$CommandResult.ExitCode
            if ($null -ne $CommandResult.PSObject.Properties["Invocation"]) {
                $InvocationDescription = [string]$CommandResult.Invocation
            }
        } else {
            throw "ANSYS_COMMAND must return an integer or an object with ExitCode"
        }
    } catch {
        $LaunchException = $_.Exception.Message
    }
    $Timer.Stop()

    $LogText = if (Test-Path -LiteralPath $AnsysLog) { [System.IO.File]::ReadAllText($AnsysLog) } else { "" }
    $RunCompleted = $LogText -match "ANSYS RUN COMPLETED"
    $ZeroErrors = $LogText -match "NUMBER OF ERROR\s+MESSAGES ENCOUNTERED\s*=\s*0"
    $LicenseError = $LogText -match "(?i)(ANSYS LICENSE MANAGER ERROR|license not available|ANSYSLI exited)"
    $ModalValues = @(Get-ModalValues $ModalFile)
    $ValidModes = ($ModalValues.Count -eq 20)
    if ($ValidModes) {
        foreach ($Value in $ModalValues) {
            if (-not (Test-FinitePositive $Value)) { $ValidModes = $false; break }
        }
    }

    if ($null -ne $LaunchException) {
        $SolverStatus = "LAUNCH_EXCEPTION_NO_RETRY"
        $InvocationDescription = "EXCEPTION: $LaunchException"
    } elseif (($CommandExitCode -eq 0) -and $RunCompleted -and $ZeroErrors -and (-not $LicenseError) -and $ValidModes) {
        $SolverStatus = "SUCCESS"
    } else {
        $SolverStatus = "FAILED_NO_RETRY"
    }

    $Record.process_exit_code = if ($null -eq $CommandExitCode) { "" } else { [string]$CommandExitCode }
    $Record.runtime_s = $Timer.Elapsed.TotalSeconds.ToString("R", [System.Globalization.CultureInfo]::InvariantCulture)
    $Record.solver_status = $SolverStatus
    $Record.run_completed_marker = $RunCompleted.ToString().ToLowerInvariant()
    $Record.zero_error_messages_marker = $ZeroErrors.ToString().ToLowerInvariant()
    $Record.license_error_detected = $LicenseError.ToString().ToLowerInvariant()
    $Record.modal_frequency_count = [string]$ModalValues.Count
    $Record.output_modal_freqs_sha256 = Get-Sha256 $ModalFile
    $Record.ans_log_sha256 = Get-Sha256 $AnsysLog
    $Record.invocation = $InvocationDescription
    if ($ModalValues.Count -eq 20) {
        for ($Mode = 1; $Mode -le 20; $Mode++) {
            $Record[("fe_f{0:D2}_hz" -f $Mode)] = $ModalValues[$Mode - 1].ToString("R", [System.Globalization.CultureInfo]::InvariantCulture)
        }
    }

    $AttemptResult = [ordered]@{}
    foreach ($Key in $Record.Keys) { $AttemptResult[$Key] = $Record[$Key] }
    $AttemptResult["finished_at"] = [DateTimeOffset]::Now.ToString("o")
    $AttemptResult["no_retry_performed"] = $true
    Write-JsonNoBom $AttemptResultPath $AttemptResult

    $Records.Add($Record)
    Write-CumulativeOutputs $Records $false
    Write-Host ("[{0}/50] {1}: {2}" -f ($Index + 1), $CaseName, $SolverStatus)
}

Write-CumulativeOutputs $Records $true
$FinalStatus = Get-Content -Raw -LiteralPath $TopStatusPath | ConvertFrom-Json
Write-Host ("CASE_C_HOST_RUN={0}; valid={1}/50; retries=0; replacements=0" -f $FinalStatus.CASE_C_HOST_RUN, $FinalStatus.valid_20_mode_outputs)
if ($FinalStatus.CASE_C_HOST_RUN -ne "PASS") { exit 30 }
exit 0

