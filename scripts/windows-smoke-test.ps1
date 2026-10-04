<#
    Inbound Surveillance - Windows install + launch smoke test.

    Runs on a real Windows host (GitHub Actions windows-latest).
    Installs the shipped NSIS package silently, boots the Python engine,
    exercises the HTTP API, launches the Tauri GUI, then uninstalls.

    Every assertion emits a PASS/FAIL line. Exits 1 if any check fails.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string] $Installer,
    [string] $ExpectedVersion = "0.1.5",
    [int]    $EngineBootTimeoutSec = 240,
    [int]    $GuiSoakSec = 60,
    # Optional customer config to deploy before boot. When supplied, the test
    # asserts the running engine actually honoured it (workplace_type, etc).
    [string] $ConfigFile = ""
)

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"

$script:Results = New-Object System.Collections.Generic.List[object]
$script:Failures = 0

function Write-Step { param([string]$m) Write-Host "`n=== $m ===" -ForegroundColor Cyan }

function Assert-That {
    param(
        [Parameter(Mandatory = $true)][string] $Name,
        [Parameter(Mandatory = $true)][bool]   $Condition,
        [string] $Detail = ""
    )
    if ($Condition) {
        $script:Results.Add([pscustomobject]@{ Check = $Name; Result = "PASS"; Detail = $Detail })
        Write-Host ("  [PASS] {0}{1}" -f $Name, $(if ($Detail) { " - $Detail" })) -ForegroundColor Green
    }
    else {
        $script:Failures++
        $script:Results.Add([pscustomobject]@{ Check = $Name; Result = "FAIL"; Detail = $Detail })
        Write-Host ("  [FAIL] {0}{1}" -f $Name, $(if ($Detail) { " - $Detail" })) -ForegroundColor Red
    }
}

function Get-PeMachine {
    param([string] $Path)
    try {
        $fs = [System.IO.File]::OpenRead($Path)
        $br = New-Object System.IO.BinaryReader($fs)
        $fs.Position = 0x3C
        $peOff = $br.ReadInt32()
        $fs.Position = $peOff + 4
        $machine = $br.ReadUInt16()
        $br.Close(); $fs.Close()
        switch ($machine) {
            0x8664 { return "x64" }
            0x014c { return "x86" }
            0xAA64 { return "arm64" }
            default  { return ("0x{0:X}" -f $machine) }
        }
    } catch { return "unreadable" }
}

$artifactDir = Split-Path -Parent $Installer
$workDir = Join-Path $env:RUNNER_TEMP "smoke"
New-Item -ItemType Directory -Force -Path $workDir | Out-Null

$installDir = Join-Path $env:LOCALAPPDATA "Inbound Surveillance"
$engineExe  = Join-Path $installDir "inbound-engine.exe"
$appExe     = Join-Path $installDir "inbound-surveillance.exe"

Write-Host "Installer      : $Installer"
Write-Host "Install target : $installDir"
Write-Host "Expected ver   : $ExpectedVersion"

# ---------------------------------------------------------------- pre-flight
Write-Step "Pre-flight"
Assert-That "installer file exists"        (Test-Path $Installer) $Installer
Assert-That "installer is a PE executable" ((Get-PeMachine $Installer) -eq "x86" -or (Get-PeMachine $Installer) -eq "x64") ("machine=" + (Get-PeMachine $Installer))

# ---------------------------------------------------------------- uninstall any prior copy
Write-Step "Clean slate"
if (Test-Path (Join-Path $installDir "uninstall.exe")) {
    Write-Host "  running existing uninstaller"
    Start-Process -FilePath (Join-Path $installDir "uninstall.exe") -ArgumentList "/S" -Wait
    Start-Sleep -Seconds 8
}
if (Test-Path $installDir) {
    Remove-Item -Recurse -Force $installDir -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 3
}
Assert-That "install dir absent before install" (-not (Test-Path $installDir)) $installDir

# ---------------------------------------------------------------- install
Write-Step "Silent install (/S)"
$sw = [System.Diagnostics.Stopwatch]::StartNew()
try {
    $p = Start-Process -FilePath $Installer -ArgumentList "/S" -PassThru -Wait -ErrorAction Stop
    $installExit = $p.ExitCode
} catch {
    $installExit = -1
    Write-Host "  installer threw: $($_.Exception.Message)"
}
$sw.Stop()
Write-Host ("  installer exit={0} elapsed={1}s" -f $installExit, [int]$sw.Elapsed.TotalSeconds)
Assert-That "installer exited 0" ($installExit -eq 0) "exit=$installExit"
Assert-That "install dir created" (Test-Path $installDir) $installDir

# NSIS may return before files settle.
for ($i = 0; $i -lt 30 -and -not (Test-Path $appExe); $i++) { Start-Sleep -Seconds 2 }

# ---------------------------------------------------------------- payload
Write-Step "Installed payload"
Assert-That "app shell present"      (Test-Path $appExe)    "inbound-surveillance.exe"
Assert-That "engine sidecar present" (Test-Path $engineExe) "inbound-engine.exe"
Assert-That "engine is x64 PE"       ((Get-PeMachine $engineExe) -eq "x64") ("machine=" + (Get-PeMachine $engineExe))
Assert-That "app shell is x64 PE"    ((Get-PeMachine $appExe) -eq "x64") ("machine=" + (Get-PeMachine $appExe))

$vcRedist = Join-Path $installDir "resources\vc_redist.x64.exe"
$debugBat = Join-Path $installDir "resources\run-debug.bat"
Assert-That "VC++ redist bundled" (Test-Path $vcRedist) $(if (Test-Path $vcRedist) { [math]::Round((Get-Item $vcRedist).Length / 1MB, 1).ToString() + " MB" } else { "missing" })
Assert-That "run-debug.bat shipped" (Test-Path $debugBat) ""

if (Test-Path $appExe) {
    $vi = (Get-Item $appExe).VersionInfo
    Write-Host "  app version info: file=$($vi.FileVersion) product=$($vi.ProductVersion)"
    Assert-That "app shell reports $ExpectedVersion" ($vi.ProductVersion -like "*$ExpectedVersion*") "product=$($vi.ProductVersion)"
}

# sidecar must be a frozen one-file bundle, not a launcher script
$engBytes = [System.IO.File]::ReadAllBytes($engineExe)[0..1]
Assert-That "engine is a compiled binary (MZ header)" ($engBytes[0] -eq 0x4D -and $engBytes[1] -eq 0x5A) ""

# ---------------------------------------------------------------- VC++ runtime
Write-Step "Visual C++ runtime present on host"
$vcKey = "HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64"
$vcInstalled = $false
if (Test-Path $vcKey) { $vcInstalled = ((Get-ItemProperty $vcKey -ErrorAction SilentlyContinue).Installed -eq 1) }
Write-Host "  host VC++ x64 installed = $vcInstalled (runner image ships it; customer gets it via installer hook)"

# ------------------------------------------------------- customer config deploy
$dataDir = Join-Path $env:APPDATA "Inbound Surveillance"
if ($ConfigFile -and (Test-Path $ConfigFile)) {
    Write-Step "Deploy customer config"
    New-Item -ItemType Directory -Force -Path $dataDir | Out-Null
    $dest = Join-Path $dataDir "config.yaml"
    Copy-Item $ConfigFile $dest -Force
    Assert-That "customer config deployed to %APPDATA%" (Test-Path $dest) $dest

    # Reject workplace_type values the app would silently coerce to "garage".
    $rawCfg = Get-Content $dest -Raw
    $wp = $null
    if ($rawCfg -match '(?m)^\s*workplace_type:\s*(\S+)') { $wp = $Matches[1].Trim().Trim('"').Trim("'").ToLower() }
    Assert-That "config workplace_type is a valid id (garage|massage)" ($wp -in @("garage","massage")) "workplace_type='$wp'"

    # /api/config is behind _authorized(), which reads the bearer token from
    # session.json and caches it on first use. Seed it before boot so the
    # endpoint is reachable for the effective-config assertions below.
    $script:SessionToken = "smoke-test-" + [guid]::NewGuid().ToString("N")
    $session = @{ session = @{ access_token = $script:SessionToken; user = @{ id = "smoke-test" } } } |
               ConvertTo-Json -Depth 5
    Set-Content -Path (Join-Path $dataDir "session.json") -Value $session -Encoding UTF8
    Assert-That "session token seeded for authenticated endpoints" `
        (Test-Path (Join-Path $dataDir "session.json")) (Join-Path $dataDir "session.json")
} else {
    Write-Step "Deploy customer config"
    Write-Host "  no -ConfigFile supplied; testing the bundled default config"
}

# ---------------------------------------------------------------- engine boot
Write-Step "Engine boot (no camera attached - must still serve HTTP)"
$engineOut = Join-Path $workDir "engine.out.log"
$engineErr = Join-Path $workDir "engine.err.log"
$engineProc = $null
try {
    $engineProc = Start-Process -FilePath $engineExe `
        -ArgumentList "--port", "8765", "--no-browser" `
        -RedirectStandardOutput $engineOut -RedirectStandardError $engineErr `
        -PassThru -WindowStyle Hidden -ErrorAction Stop
} catch {
    Write-Host "  failed to launch engine: $($_.Exception.Message)"
}
Assert-That "engine process launched" ($null -ne $engineProc) $(if ($engineProc) { "pid=$($engineProc.Id)" } else { "no process" })

$readyLine = ""
$deadline = (Get-Date).AddSeconds($EngineBootTimeoutSec)
while ((Get-Date) -lt $deadline) {
    if (Test-Path $engineOut) {
        $txt = Get-Content $engineOut -Raw -ErrorAction SilentlyContinue
        if ($txt -and $txt -match "\[INBOUND_SERVER_READY\][^\r\n]*") { $readyLine = $Matches[0]; break }
    }
    if ($engineProc -and $engineProc.HasExited) { break }
    Start-Sleep -Seconds 3
}

Write-Host "  ---- engine stdout ----"
if (Test-Path $engineOut) { Get-Content $engineOut | Select-Object -Last 25 | ForEach-Object { Write-Host "  | $_" } }
Write-Host "  ---- engine stderr ----"
if (Test-Path $engineErr) { Get-Content $engineErr | Select-Object -Last 25 | ForEach-Object { Write-Host "  ! $_" } }

Assert-That "engine reached INBOUND_SERVER_READY" ([bool]$readyLine) $readyLine
Assert-That "boot banner reports build $ExpectedVersion" ($readyLine -like "*build=$ExpectedVersion*") $readyLine

$bootBanner = ""
if (Test-Path (Join-Path $env:APPDATA "Inbound Surveillance\logs\startup.log")) {
    $bootBanner = (Get-Content (Join-Path $env:APPDATA "Inbound Surveillance\logs\startup.log") -Raw)
}
if (-not $bootBanner -and (Test-Path (Join-Path $installDir "inbound-surveillance.log"))) {
    $bootBanner = (Get-Content (Join-Path $installDir "inbound-surveillance.log") -Raw)
}
if ($bootBanner) {
    $bb = ($bootBanner -split "`n" | Where-Object { $_ -match "INBOUND_BOOT" } | Select-Object -Last 1)
    Write-Host "  boot banner: $bb"
    Assert-That "INBOUND_BOOT banner present" ([bool]$bb) $bb
    Assert-That "banner frozen=1 (running as bundled sidecar)" ($bb -match "frozen=1") ""
} else {
    Write-Host "  (no startup.log yet - engine may still be extracting)"
}

# ---------------------------------------------------------------- HTTP API
Write-Step "HTTP API"
function Invoke-Check {
    param([string]$Url, [string]$Token = "")
    try {
        $h = @{}
        if ($Token) { $h["Authorization"] = "Bearer $Token" }
        if ($h.Count -gt 0) { return Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 30 -Headers $h }
        return Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 30
    }
    catch {
        $code = ""
        if ($_.Exception.Response) { $code = " http=$([int]$_.Exception.Response.StatusCode)" }
        Write-Host "  request error:$code $($_.Exception.Message)"
        return $null
    }
}

$root = Invoke-Check "http://127.0.0.1:8765/"
Assert-That "GET / returns 200" ($null -ne $root -and $root.StatusCode -eq 200) $(if ($root) { "status=$($root.StatusCode)" } else { "no response" })
if ($root) {
    Assert-That "GET / serves the dashboard HTML" ($root.Content -match "<title>Inbound Surveillance</title>") ("bytes=" + $root.RawContentLength)
}

$pub = Invoke-Check "http://127.0.0.1:8765/api/public-config"
Assert-That "GET /api/public-config returns 200" ($null -ne $pub -and $pub.StatusCode -eq 200) $(if ($pub) { "status=$($pub.StatusCode)" } else { "no response" })
if ($pub) {
    try {
        $j = $pub.Content | ConvertFrom-Json
        $hasUrl = [bool]($j.PSObject.Properties.Name -contains "supabase_url" -or $j.PSObject.Properties.Name -contains "url")
        Assert-That "public-config is valid JSON" $true ("keys=" + (($j.PSObject.Properties.Name) -join ","))
    } catch {
        Assert-That "public-config is valid JSON" $false $_.Exception.Message
    }
}

# The deployed customer config must actually be in force on the live engine.
if ($ConfigFile -and (Test-Path $ConfigFile)) {
    $apiCfg = Invoke-Check "http://127.0.0.1:8765/api/config" $script:SessionToken
    if ($null -ne $apiCfg -and $apiCfg.StatusCode -eq 200) {
        $live = $apiCfg.Content | ConvertFrom-Json

        Assert-That "engine honours configured workplace_type" `
            ("$($live.workplace_type)" -eq $wp) `
            ("live='$($live.workplace_type)' expected='$wp'")

        # massage must yield massage zones, not the garage fallback set
        $expectedZones = if ($wp -eq "massage") { @("treatment_room","reception") } else { @("parking","waiting") }
        $liveKinds = @()
        if ($live.PSObject.Properties.Name -contains "zone_kinds") { $liveKinds = @($live.zone_kinds) }
        if ($liveKinds.Count -gt 0) {
            $ok = $true
            foreach ($z in $expectedZones) { if ($liveKinds -notcontains $z) { $ok = $false } }
            Assert-That "workplace '$wp' produced the right zone kinds" $ok ("live=" + ($liveKinds -join ","))
        }

        Assert-That "engine honours pose_engine" ("$($live.pose_engine)" -eq "yolo") "live='$($live.pose_engine)'"

        if ($live.PSObject.Properties.Name -contains "enable_face_id") {
            Assert-That "engine honours enable_face_id=false" ($live.enable_face_id -eq $false) "live='$($live.enable_face_id)'"
        }
        if ($live.PSObject.Properties.Name -contains "store_customer_avatars") {
            Assert-That "engine honours store_customer_avatars=false" ($live.store_customer_avatars -eq $false) "live='$($live.store_customer_avatars)'"
        }

        # Belt-and-braces: no customer face data may have been written.
        $facesDir = Join-Path $dataDir "faces"
        $faceFiles = @(Get-ChildItem $facesDir -Recurse -File -ErrorAction SilentlyContinue)
        Assert-That "no customer face images written to disk" ($faceFiles.Count -eq 0) "$($faceFiles.Count) file(s) in $facesDir"
    } else {
        Assert-That "GET /api/config returns 200" $false $(if ($apiCfg) { "status=$($apiCfg.StatusCode)" } else { "no response" })
    }
}

# no-camera must not be fatal
$stillAlive = $false
if ($engineProc) { $engineProc.Refresh(); $stillAlive = -not $engineProc.HasExited }
Assert-That "engine survived with no camera attached" $stillAlive ""

# ---------------------------------------------------------------- GUI launch
Write-Step "Tauri GUI launch + soak"

# The manual engine from the previous step is still running. It would
# satisfy the sidecar-spawn and port checks below on its own, so stop it
# and confirm the port is genuinely free first.
Get-Process -Name "inbound-engine" -ErrorAction SilentlyContinue |
    Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 4
$preListen = Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue
Assert-That "port 8765 free before GUI launch" ($null -eq $preListen) ""
$preKids = Get-CimInstance Win32_Process -Filter "Name='inbound-engine.exe'" -ErrorAction SilentlyContinue
Assert-That "no engine running before GUI launch" ($null -eq $preKids) ""

$appProc = $null
try {
    $appProc = Start-Process -FilePath $appExe -PassThru -ErrorAction Stop
} catch {
    Write-Host "  failed to launch GUI: $($_.Exception.Message)"
}
Assert-That "GUI process launched" ($null -ne $appProc) $(if ($appProc) { "pid=$($appProc.Id)" } else { "no process" })

if ($appProc) {
    Write-Host "  soaking ${GuiSoakSec}s (WebView2 init + sidecar spawn + window render)"
    Start-Sleep -Seconds $GuiSoakSec
    $appProc.Refresh()
    Assert-That "GUI still alive after soak (no crash)" (-not $appProc.HasExited) ""
    if ($appProc.HasExited) { Write-Host "  GUI exit code = $($appProc.ExitCode)" }

    $win = Get-Process -Id $appProc.Id -ErrorAction SilentlyContinue
    if ($win) {
        Write-Host "  GUI MainWindowHandle=$($win.MainWindowHandle)  MainWindowTitle='$($win.MainWindowTitle)'"
        Assert-That "GUI created a top-level window" ($win.MainWindowHandle -ne 0) "title=$($win.MainWindowTitle)"
    }

    # The shell must spawn the engine sidecar as its own child process.
    # PyInstaller one-file binaries run as a bootloader that then execs the
    # real interpreter as a second process of the same name, so the process
    # holding the port is usually a grandchild of the GUI, not a direct child.
    $kids = @(Get-CimInstance Win32_Process -Filter "Name='inbound-engine.exe'" -ErrorAction SilentlyContinue)
    $mine = @($kids | Where-Object { $_.ParentProcessId -eq $appProc.Id })
    Assert-That "GUI spawned the engine as its child process" ($mine.Count -gt 0) `
        $(if ($mine.Count -gt 0) { "child pid=" + (($mine.ProcessId) -join ",") + " parent=" + $appProc.Id } else { "found $($kids.Count) engine(s) but none parented by pid $($appProc.Id)" })

    # Walk the parent chain so a PyInstaller bootloader->child pair still counts.
    # NOTE: do not name the parameter $Pid - that is a read-only automatic
    # variable in PowerShell and shadowing it breaks the walk.
    function Test-DescendantOf {
        param([int] $ProcessIdToCheck, [int] $AncestorPid)
        $seen = 0
        $current = $ProcessIdToCheck
        $chain = @($ProcessIdToCheck)
        while ($current -gt 0 -and $seen -lt 8) {
            if ($current -eq $AncestorPid) {
                Write-Host ("    ancestry {0}: {1} -> MATCH gui {2}" -f $ProcessIdToCheck, ($chain -join "->"), $AncestorPid)
                return $true
            }
            $procInfo = Get-CimInstance Win32_Process -Filter "ProcessId=$current" -ErrorAction SilentlyContinue
            if (-not $procInfo) { break }
            $current = [int]$procInfo.ParentProcessId
            $chain += $current
            $seen++
        }
        Write-Host ("    ancestry {0}: {1} -> no match (gui {2})" -f $ProcessIdToCheck, ($chain -join "->"), $AncestorPid)
        return $false
    }

    # The port must be held by an engine that descends from the GUI process.
    $listen = @(Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue)
    $enginePids = @($kids | ForEach-Object { [int]$_.ProcessId })
    $attributed = @($listen | Where-Object {
        $owner = [int]$_.OwningProcess
        ($enginePids -contains $owner) -and (Test-DescendantOf -ProcessIdToCheck $owner -AncestorPid $appProc.Id)
    })
    Assert-That "port 8765 held by an engine descended from the GUI" ($attributed.Count -gt 0) `
        $(if ($listen.Count -gt 0) {
            "listener pid=" + (($listen.OwningProcess) -join ",") +
            "; engine pids=" + ($enginePids -join ",") +
            "; gui pid=" + $appProc.Id
        } else { "no listener" })

    # End-to-end: the engine the GUI started must actually answer HTTP.
    $guiRoot = Invoke-Check "http://127.0.0.1:8765/"
    Assert-That "GUI-spawned engine serves the dashboard" `
        ($null -ne $guiRoot -and $guiRoot.StatusCode -eq 200 -and $guiRoot.Content -match "<title>Inbound Surveillance</title>") `
        $(if ($guiRoot) { "status=" + $guiRoot.StatusCode + " bytes=" + $guiRoot.RawContentLength } else { "no response" })
}

# ---------------------------------------------------------------- collect logs
Write-Step "Diagnostics"
$dataDir = Join-Path $env:APPDATA "Inbound Surveillance"
if (Test-Path $dataDir) {
    Write-Host "  data dir contents:"
    Get-ChildItem $dataDir -Recurse -Depth 2 -ErrorAction SilentlyContinue |
        Select-Object -First 40 | ForEach-Object { Write-Host "    $($_.FullName.Replace($dataDir,'.')) ($($_.Length)b)" }
    Assert-That "writable data dir created under %APPDATA%" (Test-Path $dataDir) $dataDir
    $cfg = Join-Path $dataDir "config.yaml"
    Write-Host "  config.yaml present = $(Test-Path $cfg)"
} else {
    Write-Host "  no data dir yet"
}

# ---------------------------------------------------------------- teardown
Write-Step "Teardown"
foreach ($proc in @($appProc, $engineProc)) {
    if ($proc) { try { $proc.Refresh(); if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue } } catch {} }
}
Get-Process -Name "inbound-engine", "inbound-surveillance" -ErrorAction SilentlyContinue |
    Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3
Assert-That "all app processes stopped" ($null -eq (Get-Process -Name "inbound-engine", "inbound-surveillance" -ErrorAction SilentlyContinue)) ""

# ---------------------------------------------------------------- uninstall
Write-Step "Silent uninstall"
$uninstaller = Join-Path $installDir "uninstall.exe"
if (Test-Path $uninstaller) {
    try { Start-Process -FilePath $uninstaller -ArgumentList "/S" -Wait -ErrorAction Stop } catch { Write-Host "  uninstaller error: $($_.Exception.Message)" }
    for ($i = 0; $i -lt 30 -and (Test-Path $appExe); $i++) { Start-Sleep -Seconds 2 }
    Assert-That "uninstaller removed the app shell" (-not (Test-Path $appExe)) ""
    Assert-That "uninstaller removed the engine"    (-not (Test-Path $engineExe)) ""
} else {
    Assert-That "uninstaller present" $false "uninstall.exe not found in $installDir"
}

# ---------------------------------------------------------------- summary
Write-Step "SUMMARY"
$script:Results | Format-Table -AutoSize | Out-String | Write-Host
$pass = @($script:Results | Where-Object { $_.Result -eq "PASS" }).Count
$fail = @($script:Results | Where-Object { $_.Result -eq "FAIL" }).Count
Write-Host ("CHECKS: {0} passed, {1} failed, {2} total" -f $pass, $fail, $script:Results.Count)

$script:Results | Export-Csv -NoTypeInformation -Path (Join-Path $workDir "smoke-results.csv")

if ($fail -gt 0) {
    Write-Host "SMOKE TEST FAILED" -ForegroundColor Red
    exit 1
}
Write-Host "SMOKE TEST PASSED - $pass/$($script:Results.Count)" -ForegroundColor Green
exit 0
