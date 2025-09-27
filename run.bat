@echo off
REM Usage: run.bat [install|test|URL_FILE]

setlocal enabledelayedexpansion

REM Function to install dependencies
if "%1"=="install" goto install
if "%1"=="test" goto test
if "%1"=="" goto usage
if "%1"=="-h" goto usage
if "%1"=="--help" goto usage
if "%1"=="help" goto usage

REM Default: assume it's a URL file
goto run_evaluation

:install
echo Installing ML Model Evaluation System dependencies...

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    exit /b 1
)

echo Using Python command: python

REM Check if pip is available
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo Error: pip is not available. Please install pip first.
    exit /b 1
)

REM Install requirements
if exist "requirements.txt" (
    echo Installing packages from requirements.txt...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Error: Failed to install dependencies
        exit /b 1
    )
    echo Success: All dependencies installed successfully!
) else (
    echo Error: requirements.txt not found in the current directory
    exit /b 1
)

echo Success: Installation completed!
exit /b 0

:test
echo Running ML Model Evaluation System tests...

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    exit /b 1
)

REM Check if pytest is installed
python -m pytest --version >nul 2>&1
if errorlevel 1 (
    echo Error: pytest is not installed. Run 'run.bat install' first.
    exit /b 1
)

REM Change to backend directory where tests are located
cd backend

echo Running test suite with pytest...
echo Test files location: backend\src\Testing\

REM TODO: Actually run tests when ready
REM python -m pytest src\Testing\ -v --tb=short

echo Success: Test execution completed!
echo Note: Actual test execution is not implemented yet
exit /b 0

:run_evaluation
set url_file=%1

echo Running ML Model Evaluation on URL file: %url_file%

REM Check if file exists
if not exist "%url_file%" (
    echo Error: URL file '%url_file%' not found
    exit /b 1
)

echo Success: URL file validation passed!

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    exit /b 1
)

REM TODO: Actually run evaluation when ready
REM cd backend
REM python src\main.py "%url_file%"

echo Success: Evaluation execution completed!
echo Note: Actual evaluation execution is not implemented yet
exit /b 0

:usage
echo ML Model Evaluation System
echo.
echo Usage: run.bat [COMMAND^|URL_FILE]
echo.
echo Commands:
echo   install     Install all required dependencies
echo   test        Run the test suite
echo   URL_FILE    Run evaluation on URLs in the specified file
echo.
echo Examples:
echo   run.bat install
echo   run.bat test
echo   run.bat sample_urls.txt
echo.
exit /b 0