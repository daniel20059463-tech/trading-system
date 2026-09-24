# health_check.ps1 - 交易系統每日健康檢查
# 檢查三個每日排程今天是否正常執行、log 與 wiki 是否更新，
# 輸出健康報告到月度健康 log + 桌面狀態檔，異常時桌面跳通知。
# 由 Windows 工作排程器每日 16:10 執行，或手動 / 由 Claude 隨時呼叫。
# 注意：本檔須存成 UTF-8 BOM，否則 PowerShell 5.1 會以 Big5 解讀導致亂碼。

$ErrorActionPreference = "Continue"
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$ProjectDir = $PSScriptRoot
$WikiDir    = "D:\trading-wiki"
$Today      = Get-Date -Format "yyyy-MM-dd"
$TodayDate  = (Get-Date).Date
$Now        = Get-Date -Format "yyyy-MM-dd HH:mm"

$TaskOrder  = @("PreMarket","Intraday","PostMarket")
$Suffix     = @{ "PreMarket"="pre"; "Intraday"="intraday"; "PostMarket"="post" }
$ZhName     = @{ "PreMarket"="盤前"; "Intraday"="盤中"; "PostMarket"="盤後" }

# 台股 2026 國定假日（休市日，更新年度時補上即可）
$Holidays2026 = @("2026-01-01","2026-02-16","2026-02-17","2026-02-18","2026-02-19","2026-02-20",
                  "2026-02-27","2026-02-28","2026-04-03","2026-04-06","2026-05-01","2026-06-19",
                  "2026-09-25","2026-10-09","2026-10-26","2026-12-25")

# 非交易日（週末或國定假日）→ 系統正常休息，不報異常
$dow = (Get-Date).DayOfWeek
$isNonTradingDay = ($dow -eq "Saturday") -or ($dow -eq "Sunday") -or ($Holidays2026 -contains $Today)
if ($isNonTradingDay) {
    $reason = if ($dow -eq "Saturday" -or $dow -eq "Sunday") { "週末" } else { "國定假日" }
    Write-Host ""
    Write-Host "============================================"
    Write-Host "  交易系統健康檢查  $Now"
    Write-Host "============================================"
    Write-Host "  [非交易日-$reason] 台股休市，排程正常休息，無需檢查"
    Write-Host "============================================"
    $statusFile = Join-Path ([Environment]::GetFolderPath("Desktop")) "交易系統狀態.txt"
    "交易系統健康狀態   更新於 $Now`r`n============================`r`n總評：[非交易日-$reason] 台股休市，系統正常休息" | Out-File -FilePath $statusFile -Encoding utf8
    exit 0
}

$results = @()
$allOK   = $true

foreach ($task in $TaskOrder) {
    $item = [ordered]@{ task=$task; ran_today=$false; result=$null; log_ok=$false; status="FAIL"; detail="" }

    # 1) 排程執行狀況
    $info = $null
    try { $info = Get-ScheduledTaskInfo -TaskName "TradingSystem_$task" -ErrorAction Stop } catch {}
    if ($info) {
        $ranToday = ($info.LastRunTime -is [datetime]) -and ($info.LastRunTime.Date -eq $TodayDate)
        $item.ran_today = $ranToday
        $item.result    = $info.LastTaskResult
        if (-not $ranToday) {
            $item.detail = "今天未執行(上次 $($info.LastRunTime))"
        } elseif ($info.LastTaskResult -ne 0) {
            $item.detail = "有跑但回傳碼 $($info.LastTaskResult) 非0"
        }
    } else {
        $item.detail = "找不到排程 TradingSystem_$task"
    }

    # 2) 今日 log 檔存在且非空
    $logPath = Join-Path $ProjectDir "logs\${Today}_$($Suffix[$task]).log"
    if (Test-Path $logPath) {
        $sz = (Get-Item $logPath).Length
        $item.log_ok = ($sz -gt 100)
        if (-not $item.log_ok) { $item.detail += " | log 過小($sz bytes)" }
    } else {
        $item.detail += " | 今日無 log 檔"
    }

    # 3) 綜合判定
    if ($item.ran_today -and $item.result -eq 0 -and $item.log_ok) {
        $item.status = "OK"
    } elseif ($item.ran_today -and $item.result -eq 0) {
        $item.status = "WARN"
    } else {
        $item.status = "FAIL"; $allOK = $false
    }
    $results += [pscustomobject]$item
}

# 4) 今日 wiki 是否更新
$wikiPath = Join-Path $WikiDir "$Today.md"
$wikiOK = (Test-Path $wikiPath) -and ((Get-Item $wikiPath).LastWriteTime.Date -eq $TodayDate)
if (-not $wikiOK) { $allOK = $false }

$overall = if ($allOK -and $wikiOK) { "[全部正常]" } else { "[有異常需檢查]" }

function Mark($s) { switch ($s) { "OK" {"[正常]"} "WARN" {"[警告]"} default {"[異常]"} } }

# ── 主控台輸出 ──
Write-Host ""
Write-Host "============================================"
Write-Host "  交易系統健康檢查  $Now"
Write-Host "============================================"
foreach ($r in $results) {
    $line = "$(Mark $r.status) $($ZhName[$r.task])  今日執行:$(if($r.ran_today){'是'}else{'否'})  回傳:$($r.result)  log:$(if($r.log_ok){'OK'}else{'X'})"
    if ($r.detail) { $line += "  -> $($r.detail)" }
    Write-Host $line
}
Write-Host "$(if($wikiOK){'[正常]'}else{'[異常]'}) Wiki 今日記錄: $(if($wikiOK){'已更新'}else{'未更新'})"
Write-Host "--------------------------------------------"
Write-Host "  總評：$overall"
Write-Host "============================================"

# ── 月度健康 log（每日一行）──
$healthLog = Join-Path $WikiDir ("health_" + (Get-Date -Format "yyyy-MM") + ".md")
if (-not (Test-Path $healthLog)) {
    "# 系統健康記錄 $(Get-Date -Format 'yyyy-MM')`n`n| 日期 | 盤前 | 盤中 | 盤後 | Wiki | 總評 |`n|------|------|------|------|------|------|" | Out-File -FilePath $healthLog -Encoding utf8
}
$cells = $results | ForEach-Object { Mark $_.status }
$row = "| $Today | $($cells[0]) | $($cells[1]) | $($cells[2]) | $(if($wikiOK){'[正常]'}else{'[異常]'}) | $overall |"
$existing = Get-Content $healthLog -Encoding UTF8
if ($existing -match "^\| $Today \|") {
    ($existing | ForEach-Object { if ($_ -match "^\| $Today \|") { $row } else { $_ } }) | Out-File -FilePath $healthLog -Encoding utf8
} else {
    Add-Content -Path $healthLog -Value $row -Encoding utf8
}

# ── 桌面狀態檔（回家一眼看到）──
$statusFile = Join-Path ([Environment]::GetFolderPath("Desktop")) "交易系統狀態.txt"
$body = "交易系統健康狀態   更新於 $Now`r`n============================`r`n總評：$overall`r`n`r`n"
foreach ($r in $results) { $body += "$($ZhName[$r.task]): $(Mark $r.status)$(if($r.detail){' - '+$r.detail})`r`n" }
$body += "Wiki 今日記錄：$(if($wikiOK){'已更新'}else{'未更新'})`r`n`r`n(此檔每日 16:10 自動更新；[異常] 代表有任務沒跑成功，需檢查)"
$body | Out-File -FilePath $statusFile -Encoding utf8

# ── 異常時跳桌面通知 ──
if (-not ($allOK -and $wikiOK)) {
    try {
        Add-Type -AssemblyName System.Windows.Forms
        $balloon = New-Object System.Windows.Forms.NotifyIcon
        $balloon.Icon = [System.Drawing.SystemIcons]::Warning
        $balloon.BalloonTipTitle = "交易系統健康檢查"
        $balloon.BalloonTipText  = "今日有排程未正常執行，請查看桌面『交易系統狀態.txt』"
        $balloon.Visible = $true
        $balloon.ShowBalloonTip(10000)
        Start-Sleep -Seconds 1
        $balloon.Dispose()
    } catch {}
}

if ($allOK -and $wikiOK) { exit 0 } else { exit 1 }
