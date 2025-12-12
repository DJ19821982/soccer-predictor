
# mvp_predictor.py
# Minimal MVP pipeline for soccer: fetch (football-data.org or local openfootball files),
# build simple Elo + Poisson model, and predict upcoming fixtures.
#
# Notes:
# - To use football-data.org API, set the environment variable FOOTBALL_DATA_API_KEY
#   or pass it to the functions.
# - Alternatively, place openfootball JSON files (or CSV) in data/openfootball/
#   and the loader will read them. See README for format details.
#
import os, math, sqlite3, time, json
from datetime import datetime
from collections import defaultdict
import requests
import numpy as np
from scipy.stats import poisson

DB_FILE = 'matches.db'

class DataStore:
    def __init__(self, path=DB_FILE):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._ensure()

    def _ensure(self):
        c = self.conn.cursor()
        c.execute('''
        CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            competition TEXT,
            season INTEGER,
            home_team TEXT,
            away_team TEXT,
            home_goals INTEGER,
            away_goals INTEGER
        )''')
        self.conn.commit()

    def add_match(self, date, competition, season, home, away, hg, ag):
        c = self.conn.cursor()
        c.execute('''INSERT INTO matches (date, competition, season, home_team, away_team, home_goals, away_goals)
                     VALUES (?, ?, ?, ?, ?, ?, ?)''', (date, competition, season, home, away, hg, ag))
        self.conn.commit()

    def list_matches(self, competition=None, season=None, limit=None):
        c = self.conn.cursor()
        q = 'SELECT date,competition,season,home_team,away_team,home_goals,away_goals FROM matches WHERE home_goals IS NOT NULL'\n            params = []
        if competition:
            q += ' AND competition=?'; params.append(competition)
        if season:
            q += ' AND season=?'; params.append(season)
        q += ' ORDER BY date ASC'
        if limit:
            q += f' LIMIT {limit}'
        c.execute(q, params)
        return c.fetchall()

# ------------------- Loaders -------------------
def fetch_from_football_data(competition_code, season=None, api_key=None):
    \"\"\"Fetch matches for competition_code (like 'PL' for Premier League) from football-data.org.
       Requires API Key (free tier). Returns list of matches in a simplified format.\n        \"\"\"\n        if api_key is None:
        api_key = os.getenv('FOOTBALL_DATA_API_KEY')\n        headers = {'X-Auth-Token': api_key} if api_key else {}\n        url = f'https://api.football-data.org/v4/competitions/{competition_code}/matches'\n        params = {}\n        if season:
        params['season'] = season\n        r = requests.get(url, headers=headers, params=params, timeout=30)\n        r.raise_for_status()\n        j = r.json()\n        out = []\n        for m in j.get('matches', []):\n            utc = m.get('utcDate') or m.get('utcDate', None)\n            date = utc[:10] if utc else None\n            comp = m.get('competition', {}).get('code', competition_code)\n            s = m.get('season', {}).get('startDate', None)\n            season_year = None\n            try:\n                if s:\n                    season_year = int(s[:4])\n            except:\n                season_year = None\n            home = m.get('homeTeam', {}).get('name')\n            away = m.get('awayTeam', {}).get('name')\n            hg = m.get('score', {}).get('fullTime', {}).get('home')\n            ag = m.get('score', {}).get('fullTime', {}).get('away')\n            out.append((date, comp, season_year, home, away, hg, ag))\n        return out\n\n    def load_openfootball_json(folder):\n        \"\"\"Load matches from local JSON files exported from openfootball or similar sources.\n           Files should be newline-delimited JSON objects with keys: date, competition, season, home, away, home_goals, away_goals\n        \"\"\"\n        out = []\n        if not os.path.isdir(folder):\n            return out\n        for fname in os.listdir(folder):\n            path = os.path.join(folder, fname)\n            if not os.path.isfile(path):\n                continue\n            try:\n                with open(path, 'r', encoding='utf-8') as f:\n                    for line in f:\n                        line=line.strip()\n                        if not line: continue\n                        j = json.loads(line)\n                        out.append((j.get('date'), j.get('competition'), j.get('season'), j.get('home'), j.get('away'), j.get('home_goals'), j.get('away_goals')))\n            except Exception as e:\n                print('skip', path, e)\n        return out\n\n    # ------------------- Simple Elo -------------------\n    class EloRatings:\n        def __init__(self, k=20, base=1500):\n            self.k = k\n            self.ratings = defaultdict(lambda: base)\n\n        def expected(self, a, b):\n            ra = self.ratings[a]; rb = self.ratings[b]\n            ea = 1.0/(1+10**((rb-ra)/400.0))\n            return ea\n\n        def update(self, a, b, score_a, score_b):\n            ea = self.expected(a,b)\n            if score_a > score_b:\n                sa = 1.0\n            elif score_a == score_b:\n                sa = 0.5\n            else:\n                sa = 0.0\n            self.ratings[a] += self.k*(sa - ea)\n            self.ratings[b] += self.k*((1-sa) - (1-ea))\n\n    # ------------------- Poisson model -------------------\n    def fit_poisson_strengths(matches):\n        \"\"\"Compute simple attack/defense strengths and league average goals.\n           matches: list of tuples (date, competition, season, home, away, hg, ag) with goals present\n        \"\"\"\n        teams = set()\n        for m in matches:\n            teams.add(m[3]); teams.add(m[4])\n        teams = list(teams)\n        attack = defaultdict(float); defense = defaultdict(float); games = defaultdict(int)\n        for m in matches:\n            hg = m[5]; ag = m[6]\n            if hg is None or ag is None:\n                continue\n            h = m[3]; a = m[4]\n            attack[h] += hg; defense[h] += ag; games[h] += 1\n            attack[a] += ag; defense[a] += hg; games[a] += 1\n        total_goals = sum(attack.values())\n        total_games = sum(games.values())\n        avg_goals = (total_goals / total_games) if total_games>0 else 1.4\n        atk = {}\n        defn = {}\n        for t in teams:\n            if games[t] > 0:\n                atk[t] = (attack[t]/games[t]) / avg_goals\n                defn[t] = (defense[t]/games[t]) / avg_goals\n            else:\n                atk[t] = 1.0; defn[t] = 1.0\n        return atk, defn, avg_goals\n\n    def predict_match_poisson(home, away, atk, defn, avg_goals, home_adv=1.05):\n        lam_h = avg_goals * atk.get(home,1.0) * defn.get(away,1.0) * home_adv\n        lam_a = avg_goals * atk.get(away,1.0) * defn.get(home,1.0)\n        max_goal = 6\n        scores = []\n        for gh in range(0, max_goal+1):\n            for ga in range(0, max_goal+1):\n                p = poisson.pmf(gh, lam_h) * poisson.pmf(ga, lam_a)\n                scores.append(((gh,ga), p))\n        scores.sort(key=lambda x: -x[1])\n        # aggregate win/draw/loss probs\n        p_win = sum(p for (s,p) in scores if s[0]>s[1])\n        p_draw = sum(p for (s,p) in scores if s[0]==s[1])\n        p_loss = sum(p for (s,p) in scores if s[0]<s[1])\n        return {\n            'lambda_home': lam_h, 'lambda_away': lam_a,\n            'top_scores': scores[:6],\n            'p_win': p_win, 'p_draw': p_draw, 'p_loss': p_loss\n        }\n\n    # ------------------- High-level workflow helpers -------------------\n    def ingest_football_data_competition(ds, competition_code, season=None, api_key=None):\n        items = fetch_from_football_data(competition_code, season=season, api_key=api_key)\n        for it in items:\n            date, comp, season_year, home, away, hg, ag = it\n            if home is None or away is None:\n                continue\n            ds.add_match(date or datetime.utcnow().date().isoformat(), comp or competition_code, season_year or 0, home, away, hg, ag)\n        return len(items)\n\n    def ingest_openfootball_folder(ds, folder):\n        items = load_openfootball_json(folder)\n        for it in items:\n            date, comp, season, home, away, hg, ag = it\n            if home is None or away is None:\n                continue\n            ds.add_match(date or datetime.utcnow().date().isoformat(), comp or 'OPEN', season or 0, home, away, hg, ag)\n        return len(items)\n\n    def build_models(ds, competition=None, season=None):\n        matches = ds.list_matches(competition=competition, season=season)\n        # match tuple: date, competition, season, home, away, hg, ag\n        # filter completed matches\n        complete = [m for m in matches if m[5] is not None and m[6] is not None]\n        atk, defn, avg = fit_poisson_strengths(complete)\n        # simple Elo\n        elo = EloRatings()\n        for m in complete:\n            h = m[3]; a = m[4]; hg = m[5]; ag = m[6]\n            elo.update(h,a,hg,ag)\n        return {'atk': atk, 'defn': defn, 'avg_goals': avg, 'elo': elo}\n\n    def predict_for_upcoming(ds, models, competition=None, season=None):\n        # upcoming: those with null goals\n        c = ds.conn.cursor()\n        q = 'SELECT date,competition,season,home_team,away_team FROM matches WHERE home_goals IS NULL'\n        params = []\n        if competition:\n            q = q.replace('WHERE','WHERE competition=? AND')\n            params.insert(0,competition)\n        c.execute(q, params)\n        rows = c.fetchall()\n        out = []\n        for r in rows:\n            date, comp, s, home, away = r\n            pred = predict_match_poisson(home, away, models['atk'], models['defn'], models['avg_goals'])\n            out.append({'date': date, 'competition': comp, 'home': home, 'away': away, 'pred': pred})\n        return out\n\n    if __name__ == '__main__':\n        print('MVP predictor module. Import and use functions in your app.')\n