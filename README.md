
# Soccer Predictor MVP (free sources)

This MVP uses **free** data sources and local files to build simple predictions for upcoming matches.
The aim is to provide a working, easy-to-run Windows application (.exe) that you can use without coding.

## How it works (short)
- Data is stored locally in `matches.db` (SQLite).
- You can ingest data from football-data.org (requires a free API key) OR load JSON files from openfootball-style datasets.
- The app builds a simple Elo rating and Poisson-based goal model and predicts upcoming fixtures.
- The GUI (app_gui.py) lets you fetch, train, and view predictions.

## Using football-data.org (optional)
1. Create a free account at https://www.football-data.org/ and get an API key.
2. Set the environment variable `FOOTBALL_DATA_API_KEY` on your Windows PC (or export in PowerShell):
   ```powershell
   setx FOOTBALL_DATA_API_KEY "your_key_here"
   ```
3. Start the app, enter competition code (e.g. PL for Premier League), season (e.g. 2023) and click Fetch.

## Using openfootball / local files (no API key required)
- Place newline-delimited JSON files in a folder. Each line should be a JSON object like:
  `{\"date\":\"2023-08-12\",\"competition\":\"PL\",\"season\":2023,\"home\":\"Manchester United FC\",\"away\":\"Chelsea FC\",\"home_goals\":1,\"away_goals\":1}`
- In the GUI click 'Load openfootball folder' and choose the folder.

## Build a Windows EXE
- Option A (local): On Windows, open PowerShell in this folder and run `.\build.ps1` (this will install dependencies and PyInstaller and output `dist\\SoccerPredictorMVP.exe`). 
- Option B (GitHub Actions): I can add a workflow to build automatically and send you the artifact.

## Notes and limitations
- This is an MVP with simple models (Elo + Poisson). It is explainable but not state-of-the-art. Adding bookmaker odds, lineups, and richer features will improve results.
- The app does not perform live in-play predictions. It predicts pre-match using the data available in the DB.

## Next steps I can do for you
- Add automatic downloaders for many leagues and scheduled updates.
- Add ML layer (LightGBM) to improve accuracy using features.
- Polish the UI and package a proper installer (Inno Setup).
- Set up GitHub Actions to build the exe and deliver it to you.
