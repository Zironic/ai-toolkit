@echo off
REM run-ui.bat - forward to UI npm scripts
REM Usage: run-ui [npm-script] [-- <args>]
REM Example: run-ui (defaults to build_and_start)
REM Example: run-ui test:caption --silent
pushd "%~dp0ui"

REM Default to build_and_start when no args provided
if "%~1"=="" (
    npm run build_and_start
) else (
    REM run specified script, pass through additional args if any
    set SCRIPT=%~1
    shift
    if "%*"=="" (
        npm run %SCRIPT%
    ) else (
        npm run %SCRIPT% -- %*
    )
)

popd
