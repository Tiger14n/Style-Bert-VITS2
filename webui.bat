chcp 65001 > NUL
@echo off

pushd %~dp0
echo Running app.py... Might take a few seconds....
venv\Scripts\python webui.py

if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )

popd
pause