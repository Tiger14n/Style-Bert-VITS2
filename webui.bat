chcp 65001 > NUL
@echo off

pushd %~dp0
echo Running webui.py... Might take a while...
venv\Scripts\python webui.py

if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )

popd
pause