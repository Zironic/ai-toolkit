@echo off
REM run-ui-dev.bat - start UI in development mode (hot reload)
REM Usage: run-ui-dev

pushd "%~dp0ui"

REM Ensure dependencies installed
npm install

REM Start dev server (includes cron worker + Next.js dev with hot reload)
npm run dev

popd
