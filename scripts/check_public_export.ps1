param(
    [string] $ExportRoot = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($ExportRoot.Length -eq 0) {
    $ExportRoot = Join-Path $scriptRoot ".."
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Label,
        [Parameter(Mandatory = $true)]
        [scriptblock] $Command
    )

    Write-Host "==> $Label"
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Label,
        [Parameter(Mandatory = $true)]
        [string] $FilePath,
        [string[]] $Arguments = @()
    )

    Write-Host "==> $Label"
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

function Copy-ExportForVerify {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Source,
        [Parameter(Mandatory = $true)]
        [string] $Destination
    )

    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    Get-ChildItem -LiteralPath $Source -Force | ForEach-Object {
        if (($_.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "reparse point is not allowed in copied public export: $($_.FullName)"
        }
        if ($_.Name -in @("target", ".git")) {
            return
        }

        $target = Join-Path $Destination $_.Name
        if ($_.PSIsContainer) {
            Copy-ExportForVerify -Source $_.FullName -Destination $target
        } else {
            Copy-Item -LiteralPath $_.FullName -Destination $target
        }
    }
}

function Assert-NoReparsePoints {
    param([Parameter(Mandatory = $true)] [string] $Root)

    Get-ChildItem -LiteralPath $Root -Force -Recurse | ForEach-Object {
        if (($_.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "reparse point is not allowed in public export: $($_.FullName)"
        }
    }
}

function Assert-TextFileIsBounded {
    param([Parameter(Mandatory = $true)] [System.IO.FileInfo] $File)

    $maxFileBytes = 2MB
    if ($File.Length -gt $maxFileBytes) {
        throw "public export file too large: $($File.FullName)"
    }
    $bytes = [System.IO.File]::ReadAllBytes($File.FullName)
    foreach ($byte in $bytes) {
        if ($byte -eq 0 -or ($byte -lt 32 -and $byte -ne 9 -and $byte -ne 10 -and $byte -ne 13)) {
            throw "control byte found in public export text file: $($File.FullName)"
        }
    }
}

function Test-PublicBinaryAssetRelative {
    param([Parameter(Mandatory = $true)] [string] $RelativePath)

    return $publicBinaryAssetSet.Contains($RelativePath)
}

function Assert-PublicFileIsBounded {
    param(
        [Parameter(Mandatory = $true)] [System.IO.FileInfo] $File,
        [Parameter(Mandatory = $true)] [string] $RelativePath
    )

    if (Test-PublicBinaryAssetRelative -RelativePath $RelativePath) {
        if ($File.Length -gt $maxPublicBinaryAssetBytes) {
            throw "public binary asset too large: $RelativePath"
        }
        return
    }

    Assert-TextFileIsBounded -File $File
}

function Assert-NoTextPattern {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Root,
        [Parameter(Mandatory = $true)]
        [string] $Pattern,
        [Parameter(Mandatory = $true)]
        [string] $Label,
        [string[]] $ExcludeRelative = @()
    )

    $scanRoot = Resolve-Path -LiteralPath $Root
    $scanRootFullName = $scanRoot.Path.TrimEnd("\", "/")
    $excludeSet = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
    foreach ($entry in $ExcludeRelative) {
        if (-not $excludeSet.Add($entry)) {
            throw "duplicate text-scan exclude entry: $entry"
        }
    }

    Get-ChildItem -LiteralPath $scanRoot -Recurse -Force -File | ForEach-Object {
        $fileFullName = $_.FullName
        if (-not $fileFullName.StartsWith($scanRootFullName, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "text scan file escapes scan root: $fileFullName"
        }

        $relative = $fileFullName.Substring($scanRootFullName.Length).TrimStart("\", "/") -replace "\\", "/"
        if (
            $relative.StartsWith(".git/", [System.StringComparison]::Ordinal) -or
            $relative.StartsWith("target/", [System.StringComparison]::Ordinal) -or
            $excludeSet.Contains($relative) -or
            (Test-PublicBinaryAssetRelative -RelativePath $relative)
        ) {
            return
        }

        Assert-TextFileIsBounded -File $_
        $content = [System.IO.File]::ReadAllText($fileFullName)
        if ($content -match $Pattern) {
            throw "$Label found in public export file: $relative"
        }
    }
}

function Assert-NoForbiddenPublicPath {
    param([Parameter(Mandatory = $true)] [string] $RelativePath)

    $normalized = $RelativePath -replace "\\", "/"
    foreach ($fragment in $forbiddenPublicPathFragments) {
        if ($normalized.IndexOf($fragment, [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
            throw "forbidden private path fragment '$fragment' in public export path: $RelativePath"
        }
    }
}

function New-OrdinalSet {
    param([Parameter(Mandatory = $true)] [string[]] $Items)

    $set = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
    foreach ($item in $Items) {
        if (-not $set.Add($item)) {
            throw "duplicate exact allowlist entry: $item"
        }
    }
    return $set
}

function Assert-ExactOrdinalSet {
    param(
        [Parameter(Mandatory = $true)]
        [System.Collections.Generic.HashSet[string]] $Actual,
        [Parameter(Mandatory = $true)]
        [System.Collections.Generic.HashSet[string]] $Expected,
        [Parameter(Mandatory = $true)]
        [string] $Label
    )

    foreach ($item in $Actual) {
        if (-not $Expected.Contains($item)) {
            throw "unexpected ${Label}: $item"
        }
    }
    foreach ($item in $Expected) {
        if (-not $Actual.Contains($item)) {
            throw "missing ${Label}: $item"
        }
    }
}

$exportPath = Resolve-Path -LiteralPath $ExportRoot
$exportFullName = $exportPath.Path.TrimEnd("\", "/")
Push-Location $exportPath
$previousCargoTargetDir = $env:CARGO_TARGET_DIR
$guardCargoTargetDir = Join-Path ([System.IO.Path]::GetTempPath()) ("alphasync_public_export_target_" + [System.Guid]::NewGuid().ToString("N"))
$generatedCargoDirName = "tar" + "get"
$generatedCargoGlob = "!" + $generatedCargoDirName + "/**"
$forbiddenPublicPathFragments = @(
    ("golden_" + "refactor"),
    ("golden-" + "refactor"),
    ("golden_" + "connector"),
    ("golden_" + "legacy_parity"),
    ("alphasync-" + "selftrain"),
    ("alphasync-" + "skillstore"),
    ("docs/" + "vn" + "gard")
)
$maxPublicBinaryAssetBytes = 4MB
$publicBinaryAssets = @(
    "docs/assets/vraxion-home-hero.jpg",
    "docs/assets/vraxion-home-hero.webp",
    "docs/assets/vraxion-wordmark.webp",
    "docs/assets/fonts/geist-sans-variable.woff2",
    "docs/instnct/assets/engine-scope-bg.jpg",
    "docs/instnct/assets/engine-scope-bg.webp",
    "docs/instnct/assets/exact-mode-bg.jpg",
    "docs/instnct/assets/exact-mode-bg.webp",
    "docs/instnct/assets/cli-proof-bg.jpg",
    "docs/instnct/assets/cli-proof-bg.webp",
    "docs/instnct/assets/constraints-founder-bg.jpg",
    "docs/instnct/assets/constraints-founder-bg.webp",
    "docs/instnct/assets/fabric-result-bg.jpg",
    "docs/instnct/assets/fabric-result-bg.webp",
    "docs/instnct/assets/instnct-hero-bg.jpg",
    "docs/instnct/assets/instnct-hero-bg.webp",
    "docs/instnct/assets/instnct-logo.webp",
    "docs/instnct/assets/proof-pack-bg.jpg",
    "docs/instnct/assets/proof-pack-bg.webp",
    "docs/instnct/assets/release-claim-bg.jpg",
    "docs/instnct/assets/release-claim-bg.webp",
    "docs/instnct/assets/t1-reflex-bg.jpg",
    "docs/instnct/assets/vraxion-note-bg.jpg",
    "docs/instnct/assets/vraxion-note-bg.webp"
)
$publicBinaryAssetSet = New-OrdinalSet -Items $publicBinaryAssets
$privatePersistenceCrate = "alphasync-" + "skillstore"
$privateSelfTrainCrate = "alphasync-" + "selftrain"
$privateGoldenCrate = "golden-" + "refactor"
$privateGoldenPath = "golden_" + "refactor"
$privateDbCrate = "red" + "b"
$privateDbModule = $privateDbCrate + "::"
$privateDependencyPattern = @(
    $privatePersistenceCrate,
    ("../" + $privatePersistenceCrate),
    $privateSelfTrainCrate,
    ("../" + $privateSelfTrainCrate),
    $privateGoldenCrate,
    $privateGoldenPath,
    ("../" + $privateGoldenPath),
    $privateDbModule
) | ForEach-Object {
    [regex]::Escape($_)
}
$privateDependencyPattern = $privateDependencyPattern -join "|"
$trainingSurfacePattern = @(
    ("BEGIN_" + "PRIVATE"),
    ("END_" + "PRIVATE"),
    ("run_logic_iq_" + "train"),
    ("LogicIq" + "TrainConfig"),
    ("LogicIq" + "TrainResult"),
    ("MAX_LOGIC_IQ_PUBLIC_" + "CANDIDATE_CASE_EVALS"),
    ("LogicIq" + "Training"),
    ("write_training_" + "partial_status"),
    ("training_partial_" + "status")
) | ForEach-Object {
    [regex]::Escape($_)
}
$trainingSurfacePattern = $trainingSurfacePattern -join "|"
$privateTextLiteralFragments = @(
    ("C:" + "\Users"),
    ("S:" + "\AI"),
    ("MESSY TRAINING" + " DATA"),
    ("Fineweb" + " edu"),
    ("OPENAI_" + "API_KEY"),
    ("ANTHROPIC_" + "API_KEY"),
    ("GITHUB_" + "TOKEN")
)
$privateTextRegexFragments = @(
    ("BEGIN " + ".*PRIVATE" + " KEY")
)
$privateTextPatternParts = @()
$privateTextPatternParts += $privateTextLiteralFragments | ForEach-Object {
    [regex]::Escape($_)
}
$privateTextPatternParts += $privateTextRegexFragments
$privateTextPattern = $privateTextPatternParts -join "|"
try {
    $exportTarget = Join-Path $exportPath $generatedCargoDirName
    if (Test-Path -LiteralPath $exportTarget) {
        throw "public export contains generated Cargo target directory before guard starts"
    }
    New-Item -ItemType Directory -Force -Path $guardCargoTargetDir | Out-Null
    $env:CARGO_TARGET_DIR = $guardCargoTargetDir

    Write-Host "==> public export root shape"
    Assert-NoReparsePoints -Root $exportPath
    $requiredRootEntries = @(
        ".github\pull_request_template.md",
        ".github\workflows\ci.yml",
        ".github\workflows\deploy-instnct-notify.yml",
        ".github\workflows\public-pages-smoke.yml",
        ".github\workflows\public-surface-audit.yml",
        ".gitattributes",
        ".gitignore",
        "CHANGELOG.md",
        "CITATION.cff",
        "CODE_OF_CONDUCT.md",
        "CONTRIBUTING.md",
        "Cargo.toml",
        "Cargo.lock",
        "DEPLOYMENT.md",
        "LICENSE",
        "LICENSE_BOUNDARY.md",
        "README.md",
        "PACKAGE_BOUNDARY.md",
        "PUBLIC_DELIVERY_MODEL.md",
        "PUBLIC_BOUNDARY.md",
        "PUBLIC_RELEASE_CHECKLIST.md",
        "SECURITY.md",
        "TRADEMARK_POLICY.md",
        "docs\.nojekyll",
        "docs\404.html",
        "docs\_redirects",
        "docs\assets\favicon.svg",
        "docs\assets\vraxion-home-hero.jpg",
        "docs\assets\vraxion-home-hero.webp",
        "docs\assets\vraxion-wordmark.webp",
        "docs\assets\fonts\geist-license.txt",
        "docs\assets\fonts\geist-sans-variable.woff2",
        "docs\ANCHORCELL_RESEARCH_BRIEF.md",
        "docs\anchorcell\anchorcell.js",
        "docs\anchorcell\anchorcell.v2.example.json",
        "docs\anchorcell\anchorcell.v2.schema.json",
        "docs\anchorcell\index.html",
        "docs\anchorcell\styles.css",
        "docs\CURRENT_CAPABILITIES.md",
        "docs\CURRENT_STATUS.md",
        "docs\INSTNCT_BENCHMARK_NOTES.md",
        "docs\instnct\assets\engine-scope-bg.jpg",
        "docs\instnct\assets\engine-scope-bg.webp",
        "docs\instnct\assets\exact-mode-bg.jpg",
        "docs\instnct\assets\exact-mode-bg.webp",
        "docs\instnct\assets\cli-proof-bg.jpg",
        "docs\instnct\assets\cli-proof-bg.webp",
        "docs\instnct\assets\constraints-founder-bg.jpg",
        "docs\instnct\assets\constraints-founder-bg.webp",
        "docs\instnct\assets\fabric-result-bg.jpg",
        "docs\instnct\assets\fabric-result-bg.webp",
        "docs\instnct\assets\instnct-hero-bg.jpg",
        "docs\instnct\assets\instnct-hero-bg.webp",
        "docs\instnct\assets\instnct-logo.webp",
        "docs\instnct\assets\proof-pack-bg.jpg",
        "docs\instnct\assets\proof-pack-bg.webp",
        "docs\instnct\assets\release-claim-bg.jpg",
        "docs\instnct\assets\release-claim-bg.webp",
        "docs\instnct\assets\t1-reflex-bg.jpg",
        "docs\instnct\assets\vraxion-note-bg.jpg",
        "docs\instnct\assets\vraxion-note-bg.webp",
        "docs\instnct\index.html",
        "docs\instnct\instnct.js",
        "docs\instnct\styles.css",
        "docs\PUBLIC_SURFACE_POLICY.md",
        "docs\VERSION.json",
        "docs\index.html",
        "workers\instnct-notify\README.md",
        "workers\instnct-notify\migrations\0001_init.sql",
        "workers\instnct-notify\src\index.mjs",
        "workers\instnct-notify\wrangler.example.jsonc",
        "scripts\audit_instnct_notify_worker.mjs",
        "scripts\audit_instnct_static_site.mjs",
        "scripts\audit_public_surface.py",
        "scripts\check_public_export.ps1",
        "scripts\smoke_instnct_notify_live.mjs",
        "scripts\smoke_public_pages_links.mjs",
        "scripts\smoke_instnct_browser.mjs",
        "scripts\sync_public_release_links.mjs",
        "crates\alphasync-core\Cargo.toml",
        "crates\alphasync-core\LICENSE",
        "crates\alphasync-core\README.md",
        "crates\alphasync-runtime\Cargo.toml",
        "crates\alphasync-runtime\LICENSE",
        "crates\alphasync-runtime\README.md"
    )
    foreach ($entry in $requiredRootEntries) {
        if (-not (Test-Path -LiteralPath (Join-Path $exportPath $entry))) {
            throw "missing required public export entry: $entry"
        }
    }
    $crateDirs = @(
        Get-ChildItem -LiteralPath (Join-Path $exportPath "crates") -Directory |
            ForEach-Object { $_.Name } |
            Sort-Object
    )
    $expectedCrateDirs = @("alphasync-core", "alphasync-runtime")
    Assert-ExactOrdinalSet `
        -Actual (New-OrdinalSet -Items $crateDirs) `
        -Expected (New-OrdinalSet -Items $expectedCrateDirs) `
        -Label "public crate directory"

    Write-Host "==> public export exact path allowlist"
    $allowedPublicFiles = @(
        ".github/pull_request_template.md",
        ".github/workflows/ci.yml",
        ".github/workflows/deploy-instnct-notify.yml",
        ".github/workflows/public-pages-smoke.yml",
        ".github/workflows/public-surface-audit.yml",
        ".gitattributes",
        ".gitignore",
        "CHANGELOG.md",
        "CITATION.cff",
        "CODE_OF_CONDUCT.md",
        "CONTRIBUTING.md",
        "Cargo.lock",
        "Cargo.toml",
        "DEPLOYMENT.md",
        "LICENSE",
        "LICENSE_BOUNDARY.md",
        "PACKAGE_BOUNDARY.md",
        "PUBLIC_DELIVERY_MODEL.md",
        "PUBLIC_BOUNDARY.md",
        "PUBLIC_RELEASE_CHECKLIST.md",
        "README.md",
        "SECURITY.md",
        "TRADEMARK_POLICY.md",
        "docs/.nojekyll",
        "docs/assets/favicon.svg",
        "docs/404.html",
        "docs/_redirects",
        "docs/assets/vraxion-home-hero.jpg",
        "docs/assets/vraxion-home-hero.webp",
        "docs/assets/vraxion-wordmark.webp",
        "docs/CURRENT_CAPABILITIES.md",
        "docs/CURRENT_STATUS.md",
        "docs/INSTNCT_BENCHMARK_NOTES.md",
        "docs/instnct/assets/engine-scope-bg.jpg",
        "docs/instnct/assets/engine-scope-bg.webp",
        "docs/instnct/assets/exact-mode-bg.jpg",
        "docs/instnct/assets/exact-mode-bg.webp",
        "docs/instnct/assets/cli-proof-bg.jpg",
        "docs/instnct/assets/cli-proof-bg.webp",
        "docs/instnct/assets/constraints-founder-bg.jpg",
        "docs/instnct/assets/constraints-founder-bg.webp",
        "docs/instnct/assets/fabric-result-bg.jpg",
        "docs/instnct/assets/fabric-result-bg.webp",
        "docs/instnct/assets/instnct-hero-bg.jpg",
        "docs/instnct/assets/instnct-hero-bg.webp",
        "docs/instnct/assets/instnct-logo.webp",
        "docs/instnct/assets/proof-pack-bg.jpg",
        "docs/instnct/assets/proof-pack-bg.webp",
        "docs/instnct/assets/release-claim-bg.jpg",
        "docs/instnct/assets/release-claim-bg.webp",
        "docs/instnct/assets/t1-reflex-bg.jpg",
        "docs/instnct/assets/vraxion-note-bg.jpg",
        "docs/instnct/assets/vraxion-note-bg.webp",
        "docs/assets/fonts/geist-license.txt",
        "docs/assets/fonts/geist-sans-variable.woff2",
        "docs/ANCHORCELL_RESEARCH_BRIEF.md",
        "docs/anchorcell/anchorcell.js",
        "docs/anchorcell/anchorcell.v2.example.json",
        "docs/anchorcell/anchorcell.v2.schema.json",
        "docs/anchorcell/index.html",
        "docs/anchorcell/styles.css",
        "docs/instnct/index.html",
        "docs/instnct/instnct.js",
        "docs/instnct/styles.css",
        "docs/PUBLIC_SURFACE_POLICY.md",
        "docs/robots.txt",
        "docs/sitemap.xml",
        "docs/VERSION.json",
        "docs/index.html",
        "scripts/audit_instnct_notify_worker.mjs",
        "scripts/audit_instnct_static_site.mjs",
        "scripts/audit_public_surface.py",
        "scripts/check_public_export.ps1",
        "scripts/smoke_instnct_notify_live.mjs",
        "scripts/smoke_public_pages_links.mjs",
        "scripts/smoke_instnct_browser.mjs",
        "scripts/sync_public_release_links.mjs",
        "workers/instnct-notify/README.md",
        "workers/instnct-notify/migrations/0001_init.sql",
        "workers/instnct-notify/src/index.mjs",
        "workers/instnct-notify/wrangler.example.jsonc",
        "crates/alphasync-core/Cargo.toml",
        "crates/alphasync-core/LICENSE",
        "crates/alphasync-core/README.md",
        "crates/alphasync-core/src/eval.rs",
        "crates/alphasync-core/src/fabric.rs",
        "crates/alphasync-core/src/fabric/confidence.rs",
        "crates/alphasync-core/src/fabric/consensus.rs",
        "crates/alphasync-core/src/fabric/error.rs",
        "crates/alphasync-core/src/fabric/evolution.rs",
        "crates/alphasync-core/src/fabric/evolution_router.rs",
        "crates/alphasync-core/src/fabric/matrix.rs",
        "crates/alphasync-core/src/fabric/prismion.rs",
        "crates/alphasync-core/src/fabric/profile.rs",
        "crates/alphasync-core/src/fabric/proposal.rs",
        "crates/alphasync-core/src/ids.rs",
        "crates/alphasync-core/src/lib.rs",
        "crates/alphasync-core/src/progress.rs",
        "crates/alphasync-runtime/Cargo.toml",
        "crates/alphasync-runtime/LICENSE",
        "crates/alphasync-runtime/README.md",
        "crates/alphasync-runtime/examples/synthetic_runtime_scene.rs",
        "crates/alphasync-runtime/examples/synthetic_runtime_smoke.rs",
        "crates/alphasync-runtime/src/artifact_json.rs",
        "crates/alphasync-runtime/src/artifact_writer.rs",
        "crates/alphasync-runtime/src/lib.rs",
        "crates/alphasync-runtime/src/logic_iq.rs",
        "crates/alphasync-runtime/src/main.rs",
        "crates/alphasync-runtime/src/synthetic.rs"
    )
    $allowedPublicFileSet = New-OrdinalSet -Items $allowedPublicFiles
    $files = Get-ChildItem -LiteralPath $exportPath -Recurse -Force -File
    $totalBytes = 0L
    $relativeFileSet = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
    foreach ($file in $files) {
        $fileFullName = $file.FullName
        if (-not $fileFullName.StartsWith($exportFullName, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "export file escapes export root: $fileFullName"
        }
        $relative = $fileFullName.Substring($exportFullName.Length).TrimStart("\", "/") -replace "\\", "/"
        if ($relative -eq ".git" -or $relative.StartsWith(".git/", [System.StringComparison]::Ordinal)) {
            continue
        }
        if (-not $relativeFileSet.Add($relative)) {
            throw "duplicate public export file path: $relative"
        }
        Assert-NoForbiddenPublicPath -RelativePath $relative
        $totalBytes += $file.Length
        Assert-PublicFileIsBounded -File $file -RelativePath $relative
        if (-not $allowedPublicFileSet.Contains($relative)) {
            throw "unexpected public export file: $relative"
        }
    }
    Assert-ExactOrdinalSet -Actual $relativeFileSet -Expected $allowedPublicFileSet -Label "public export file"
    if ($totalBytes -gt 10MB) {
        throw "public export text payload too large"
    }

    Write-Host "==> public export private text scan"
    Assert-NoTextPattern `
        -Root $exportPath `
        -Pattern $privateTextPattern `
        -Label "private text pattern"

    Write-Host "==> public export private dependency text scan"
    Assert-NoTextPattern `
        -Root $exportPath `
        -Pattern $privateDependencyPattern `
        -Label "private dependency reference" `
        -ExcludeRelative @("PACKAGE_BOUNDARY.md", "scripts/check_public_export.ps1")

    Write-Host "==> public export training surface scan"
    Assert-NoTextPattern `
        -Root (Join-Path $exportPath "crates") `
        -Pattern $trainingSurfacePattern `
        -Label "private Logic-IQ training surface"

    Invoke-NativeChecked "node check audit_instnct_static_site" "node" @("--check", "scripts/audit_instnct_static_site.mjs")
    Invoke-NativeChecked "node check audit_instnct_notify_worker" "node" @("--check", "scripts/audit_instnct_notify_worker.mjs")
    Invoke-NativeChecked "node check sync_public_release_links" "node" @("--check", "scripts/sync_public_release_links.mjs")
    Invoke-NativeChecked "node check smoke_public_pages_links" "node" @("--check", "scripts/smoke_public_pages_links.mjs")
    Invoke-NativeChecked "node check smoke_instnct_notify_live" "node" @("--check", "scripts/smoke_instnct_notify_live.mjs")
    Invoke-NativeChecked "node check smoke_instnct_browser" "node" @("--check", "scripts/smoke_instnct_browser.mjs")
    Invoke-NativeChecked "node check instnct client JS" "node" @("--check", "docs/instnct/instnct.js")
    Invoke-NativeChecked "sync public release links" "node" @("scripts/sync_public_release_links.mjs", "--check")
    Invoke-NativeChecked "INSTNCT static audit" "node" @("scripts/audit_instnct_static_site.mjs")
    Invoke-NativeChecked "INSTNCT notify Worker audit" "node" @("scripts/audit_instnct_notify_worker.mjs")
    Invoke-NativeChecked "public surface audit" "python" @("scripts/audit_public_surface.py")

    Invoke-Checked "cargo metadata locked" { cargo metadata --locked --format-version 1 | Out-Null }
    Invoke-Checked "cargo fmt locked all features" {
        cargo fmt --all -- --check
    }
    Invoke-Checked "cargo test locked all features" {
        cargo test --locked --workspace --all-features
    }
    Invoke-Checked "cargo test release locked all features" {
        cargo test --release --locked --workspace --all-features
    }
    Invoke-Checked "cargo clippy locked all features" {
        cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
    }
    Invoke-Checked "cargo doc locked all features" {
        cargo doc --locked --workspace --all-features --no-deps
    }

    Write-Host "==> final public export exact path allowlist"
    if (Test-Path -LiteralPath $exportTarget) {
        throw "public export guard created target directory inside export root"
    }
    Assert-NoReparsePoints -Root $exportPath
    $finalFiles = Get-ChildItem -LiteralPath $exportPath -Recurse -Force -File
    $finalTotalBytes = 0L
    $finalRelativeFileSet = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
    foreach ($file in $finalFiles) {
        $fileFullName = $file.FullName
        if (-not $fileFullName.StartsWith($exportFullName, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "export file escapes export root after gates: $fileFullName"
        }
        $relative = $fileFullName.Substring($exportFullName.Length).TrimStart("\", "/") -replace "\\", "/"
        if ($relative -eq ".git" -or $relative.StartsWith(".git/", [System.StringComparison]::Ordinal)) {
            continue
        }
        if (-not $finalRelativeFileSet.Add($relative)) {
            throw "duplicate final public export file path: $relative"
        }
        Assert-NoForbiddenPublicPath -RelativePath $relative
        $finalTotalBytes += $file.Length
        Assert-PublicFileIsBounded -File $file -RelativePath $relative
        if (-not $allowedPublicFileSet.Contains($relative)) {
            throw "unexpected final public export file: $relative"
        }
    }
    Assert-ExactOrdinalSet -Actual $finalRelativeFileSet -Expected $allowedPublicFileSet -Label "final public export file"
    if ($finalTotalBytes -gt 10MB) {
        throw "final public export text payload too large"
    }

    Write-Host "==> cargo tree public runtime"
    $tree = cargo tree -p alphasync-runtime --all-features --prefix none
    if ($LASTEXITCODE -ne 0) {
        throw "cargo tree failed"
    }
    $treeText = $tree -join "`n"
    if ($treeText -notmatch "alphasync-runtime v") {
        throw "runtime root missing from cargo tree"
    }
    if ($treeText -notmatch "alphasync-core v") {
        throw "core dependency missing from cargo tree"
    }
    if ($treeText -match ($privatePersistenceCrate + "|" + $privateSelfTrainCrate + "|" + $privateGoldenCrate + "|" + $privateDbCrate)) {
        throw "private dependency found in public cargo tree"
    }

    Write-Host "==> copied public source export build"
    $packageVerifyRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("alphasync_public_package_verify_" + [System.Guid]::NewGuid().ToString("N"))
    if (Test-Path -LiteralPath $packageVerifyRoot) {
        Remove-Item -LiteralPath $packageVerifyRoot -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $packageVerifyRoot | Out-Null
    $sourceCopyRoot = Join-Path $packageVerifyRoot "rust_core"
    Copy-ExportForVerify -Source $exportPath -Destination $sourceCopyRoot

    Push-Location $sourceCopyRoot
    try {
        Invoke-Checked "cargo metadata copied public export" {
            cargo metadata --locked --format-version 1 | Out-Null
        }
        Invoke-Checked "cargo test copied public export" {
            cargo test --locked --workspace --all-features
        }
    }
    finally {
        Pop-Location
        if (Test-Path -LiteralPath $packageVerifyRoot) {
            Remove-Item -LiteralPath $packageVerifyRoot -Recurse -Force
        }
    }

    Write-Host "==> unpacked crate archive rebuild"
    $archiveVerifyRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("alphasync_public_archive_verify_" + [System.Guid]::NewGuid().ToString("N"))
    $archiveTarget = Join-Path $archiveVerifyRoot "target"
    $archiveExtractRoot = Join-Path $archiveVerifyRoot "extract"
    $archiveWorkspace = Join-Path $archiveVerifyRoot "workspace"
    New-Item -ItemType Directory -Force -Path $archiveTarget | Out-Null
    New-Item -ItemType Directory -Force -Path $archiveExtractRoot | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $archiveWorkspace "crates") | Out-Null

    try {
        Invoke-Checked "cargo package alphasync-core archive" {
            cargo package -p alphasync-core --allow-dirty --no-verify --target-dir $archiveTarget | Out-Null
        }
        Invoke-Checked "create alphasync-runtime source archive" {
            $runtimeArchivePackageDir = Join-Path $archiveTarget "package"
            $runtimeArchiveStaging = Join-Path $archiveVerifyRoot "runtime_archive"
            $runtimeArchiveSource = Join-Path $runtimeArchiveStaging "alphasync-runtime-0.1.0"
            New-Item -ItemType Directory -Force -Path $runtimeArchivePackageDir | Out-Null
            New-Item -ItemType Directory -Force -Path $runtimeArchiveStaging | Out-Null
            Copy-ExportForVerify `
                -Source (Join-Path $exportPath "crates\alphasync-runtime") `
                -Destination $runtimeArchiveSource
            Push-Location $runtimeArchiveStaging
            try {
                tar -czf (Join-Path $runtimeArchivePackageDir "alphasync-runtime-0.1.0.crate") "alphasync-runtime-0.1.0"
            }
            finally {
                Pop-Location
            }
        }

        $archives = @(
            Get-ChildItem -LiteralPath (Join-Path $archiveTarget "package") -Filter "*.crate" -File |
                Sort-Object Name
        )
        $expectedArchiveNames = @(
            "alphasync-core-0.1.0.crate",
            "alphasync-runtime-0.1.0.crate"
        )
        $archiveNames = @($archives | ForEach-Object { $_.Name })
        Assert-ExactOrdinalSet `
            -Actual (New-OrdinalSet -Items $archiveNames) `
            -Expected (New-OrdinalSet -Items $expectedArchiveNames) `
            -Label "crate archive"

        foreach ($archive in $archives) {
            tar -xf $archive.FullName -C $archiveExtractRoot
            if ($LASTEXITCODE -ne 0) {
                throw "failed to unpack crate archive: $($archive.Name)"
            }
        }

        Copy-ExportForVerify `
            -Source (Join-Path $archiveExtractRoot "alphasync-core-0.1.0") `
            -Destination (Join-Path $archiveWorkspace "crates\alphasync-core")
        Copy-ExportForVerify `
            -Source (Join-Path $archiveExtractRoot "alphasync-runtime-0.1.0") `
            -Destination (Join-Path $archiveWorkspace "crates\alphasync-runtime")

        @'
[workspace]
resolver = "3"
members = [
    "crates/alphasync-core",
    "crates/alphasync-runtime",
]

[workspace.package]
edition = "2024"
license = "LicenseRef-VRAXION-Community-Source-1.0"
repository = "https://github.com/VRAXION/VRAXION"
rust-version = "1.96"

[workspace.lints.rust]
unsafe_code = "forbid"
missing_docs = "warn"

[workspace.lints.clippy]
all = "warn"
pedantic = "warn"

[patch.crates-io]
alphasync-core = { path = "crates/alphasync-core" }
'@ | Set-Content -LiteralPath (Join-Path $archiveWorkspace "Cargo.toml") -NoNewline

        Push-Location $archiveWorkspace
        try {
            Invoke-Checked "cargo generate-lockfile unpacked crate workspace" {
                cargo generate-lockfile
            }
            Invoke-Checked "cargo test unpacked crate workspace" {
                cargo test --locked --workspace --all-features
            }
        }
        finally {
            Pop-Location
        }
    }
    finally {
        if (Test-Path -LiteralPath $archiveVerifyRoot) {
            Remove-Item -LiteralPath $archiveVerifyRoot -Recurse -Force
        }
    }

    Write-Host "public_export_guard=pass"
}
finally {
    if ($null -eq $previousCargoTargetDir) {
        Remove-Item Env:CARGO_TARGET_DIR -ErrorAction SilentlyContinue
    } else {
        $env:CARGO_TARGET_DIR = $previousCargoTargetDir
    }
    if (Test-Path -LiteralPath $guardCargoTargetDir) {
        Remove-Item -LiteralPath $guardCargoTargetDir -Recurse -Force
    }
    Pop-Location
}
