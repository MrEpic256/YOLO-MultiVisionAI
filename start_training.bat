@echo off
echo ========================================
echo   YOLOv8 Training Pipeline Launcher
echo ========================================
echo.

REM Перевірка Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python 3.8+
    pause
    exit /b 1
)

echo [OK] Python detected
echo.

REM Перевірка залежностей
echo Checking dependencies...
python -c "import ultralytics" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Ultralytics not installed
    echo Installing requirements...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
)

echo [OK] All dependencies installed
echo.

REM Перевірка датасету
echo Checking dataset...
python check_dataset.py
if errorlevel 1 (
    echo [ERROR] Dataset check failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Starting YOLOv8 Training...
echo ========================================
echo.

REM Запуск навчання
python train_yolov8.py

if errorlevel 1 (
    echo.
    echo [ERROR] Training failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Training Complete!
echo ========================================
echo.
echo Would you like to test the model now? (Y/N)
set /p test_choice=

if /i "%test_choice%"=="Y" (
    python test_yolov8.py
)

echo.
echo All done! Press any key to exit...
pause >nul
