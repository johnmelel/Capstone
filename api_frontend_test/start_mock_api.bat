@echo off
echo Starting Mock Medical Agent API on http://localhost:8000
echo.
echo Behavior:
echo   - Text input: Returns "the user said: your text"
echo   - Image input: Returns "image received"  
echo   - Query "image": Returns an image from ./images/ folder
echo.
python main.py
pause
