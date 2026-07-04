param(
    [Parameter(Position = 0)]
    [string]$Command = "list",

    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$localGitBug = Join-Path $repoRoot "tools\git-bug.exe"

if (Test-Path $localGitBug) {
    $GitBug = $localGitBug
} else {
    $cmd = Get-Command git-bug -ErrorAction SilentlyContinue
    if ($null -eq $cmd) {
        throw "git-bug not found. Put git-bug.exe in tools\ or install git-bug on PATH."
    }
    $GitBug = $cmd.Source
}

Push-Location $repoRoot
try {
    switch ($Command) {
        "list" {
            & $GitBug bug --status open --by edit --direction desc --format plain
        }
        "list-all" {
            & $GitBug bug --by edit --direction desc --format plain
        }
        "list-closed" {
            & $GitBug bug --status closed --by edit --direction desc --format plain
        }
        "show" {
            if ($Rest.Count -lt 1) { throw "Usage: scripts\tickets.ps1 show <id>" }
            & $GitBug bug show $Rest[0]
        }
        "new" {
            if ($Rest.Count -lt 1) { throw "Usage: scripts\tickets.ps1 new <title> [body]" }
            $title = $Rest[0]
            $body = if ($Rest.Count -gt 1) { $Rest[1] } else { $title }
            & $GitBug bug new --title $title --message $body --non-interactive
        }
        "comment" {
            if ($Rest.Count -lt 2) { throw "Usage: scripts\tickets.ps1 comment <id> <message>" }
            & $GitBug bug comment new $Rest[0] --message $Rest[1] --non-interactive
        }
        "label" {
            if ($Rest.Count -lt 2) { throw "Usage: scripts\tickets.ps1 label <id> <label> [label...]" }
            & $GitBug bug label new $Rest[0] $Rest[1..($Rest.Count - 1)]
        }
        "close" {
            if ($Rest.Count -lt 1) { throw "Usage: scripts\tickets.ps1 close <id>" }
            & $GitBug bug status close $Rest[0]
        }
        default {
            throw "Unknown command '$Command'. Use: list, list-all, list-closed, show, new, comment, label, close."
        }
    }
} finally {
    Pop-Location
}