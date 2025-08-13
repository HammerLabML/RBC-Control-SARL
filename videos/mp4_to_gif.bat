@echo off
REM Loop over alle bestanden die op het script gesleept worden
for %%a in (%*) do (
    echo Converting %%~nxa ...
    ffmpeg -i "%%~a" -filter_complex "[0:v] fps=10,scale=280:-1:flags=lanczos,split [x][y];[x] palettegen [p];[y][p] paletteuse" "%%~dpna.gif"
)
echo.
echo Klaar! Alle bestanden zijn omgezet naar GIF.
pause
