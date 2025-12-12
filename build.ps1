
# build.ps1 - Build single-file EXE with PyInstaller
param(
    [string]$script = "app_gui.py",
    [string]$outname = "SoccerPredictorMVP.exe"
)
if (-not (Get-Command pyinstaller -ErrorAction SilentlyContinue)) {
    Write-Host "Installing PyInstaller..."
    py -3 -m pip install --user pyinstaller
}
py -3 -m pip install --user -r requirements.txt
py -3 -m PyInstaller --onefile --noconsole --name $outname $script
Write-Host "Done. The exe will be in dist/"
