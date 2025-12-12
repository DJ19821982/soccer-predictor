import os, math, sqlite3
import json
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
        c.execute('''
        INSERT INTO matches (date, competition, season, home_team, away_team, home_goals, away_goals)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (date, competition, season, home, away, hg, ag))
        self.conn.commit()

    def list_matches(self, competition=None, season=None, limit=None):
        c = self.conn.cursor()
        q = 'SELECT date,competition,season,home_team,away_team,home_goals,away_goals FROM matches WHERE home_goals IS NOT NULL'
        params = []

        if competition:
            q += ' AND competition=?'
            params.append(competition)

        if season:
            q += ' AND season=?'
            params.append(season)

        q += ' ORDER BY date ASC'

        if limit:
            q += f' LIMIT {limit}'

        c.execute(q, params)
        return c.fetchall()


# ----------------------------------------------------------
# Data ingestion from football-data.org
# ----------------------------------------------------------

def fetch_from_football_data(comp, season=None, api_key=None):
    if api_key is None:
        api_key = os.getenv("FOOTBALL_DATA_API_KEY")

    headers = {'X-Auth-Token': api_key} if api_key else {}
    url = f"https://api.football-data.org/v4/competitions/{comp}/matches"

    params = {}
    if season:
        params['season'] = season

    r = requests.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()

    items = []

    for m in r.json().get('matches', []):
        date = m.get('utcDate', '')[:10]
        home = m.get('homeTeam', {}).get('name')
        away = m.get('awayTeam', {}).get('name')
        score = m.get('score', {}).get('fullTime', {})
        hg = score.get('home')
        ag = score.get('away')

        season_year = None
        s = m.get('season', {}).get('startDate')
        if s:
            try:
                season_year = int(s[:4])
            except:
                season_year = None

        items.append((date, comp, season_year, home, away, hg, ag))

    return items


def ingest_football_data_competition(ds, comp, season=None, api_key=None):
    data = fetch_from_football_data(comp, season=season, api_key=api_key)
    for date, comp, season_year, home, away, hg, ag in data:
        if home and away:
            ds.add_match(date, comp, season_year or 0, home, away, hg, ag)
    return len(data)


# ----------------------------------------------------------
# Poisson model + Elo
# ----------------------------------------------------------

class EloRatings:
    def __init__(self, k=20, base=1500):
        self.k = k
        self.ratings = defaultdict(lambda: base)

    def expected(self, a, b):
        ra = self.ratings[a]
        rb = self.ratings[b]
        return 1 / (1 + 10 ** ((rb - ra) / 400))

    def update(self, a, b, a_goals, b_goals):
        ea = self.expected(a, b)
        if a_goals > b_goals:
            sa = 1.0
        elif a_goals == b_goals:
            sa = 0.5
        else:
            sa = 0.0
        self.ratings[a] += self.k * (sa - ea)
        self.ratings[b] += self.k * ((1 - sa) - (1 - ea))


def fit_poisson_strengths(matches):
    teams = set()
    for m in matches:
        teams.add(m[3])
        teams.add(m[4])

    attack = defaultdict(float)
    defense = defaultdict(float)
    games = defaultdict(int)

    for m in matches:
        home, away, hg, ag = m[3], m[4], m[5], m[6]
        if hg is None or ag is None:
            continue
        attack[home] += hg
        defense[home] += ag
        games[home] += 1

        attack[away] += ag
        defense[away] += hg
        games[away] += 1

    total_goals = sum(attack.values())
    total_games = sum(games.values()) if sum(games.values()) > 0 else 1
    avg_goals = total_goals / total_games

    atk = {t: (attack[t] / games[t]) / avg_goals if games[t] > 0 else 1.0 for t in teams}
    defn = {t: (defense[t] / games[t]) / avg_goals if games[t] > 0 else 1.0 for t in teams}

    return atk, defn, avg_goals


def predict_match_poisson(home, away, atk, defn, avg_goals, home_adv=1.05):
    lam_h = avg_goals * atk.get(home, 1.0) * defn.get(away, 1.0) * home_adv
    lam_a = avg_goals * atk.get(away, 1.0) * defn.get(home, 1.0)

    scores = []
    for gh in range(7):
        for ga in range(7):
            p = poisson.pmf(gh, lam_h) * poisson.pmf(ga, lam_a)
            scores.append(((gh, ga), p))

    scores.sort(key=lambda x: -x[1])

    p_win = sum(p for (s, p) in scores if s[0] > s[1])
    p_draw = sum(p for (s, p) in scores if s[0] == s[1])
    p_loss = sum(p for (s, p) in scores if s[0] < s[1])

    return {
        "lambda_home": lam_h,
        "lambda_away": lam_a,
        "top_scores": scores[:6],
        "p_win": p_win,
        "p_draw": p_draw,
        "p_loss": p_loss
    }


def build_models(ds, competition=None, season=None):
    matches = ds.list_matches(competition=competition, season=season)
    complete = [m for m in matches if m[5] is not None and m[6] is not None]

    atk, defn, avg = fit_poisson_strengths(complete)

    elo = EloRatings()
    for m in complete:
        elo.update(m[3], m[4], m[5], m[6])

    return {"atk": atk, "defn": defn, "avg_goals": avg, "elo": elo}


def predict_for_upcoming(ds, models, competition=None, season=None):
    c = ds.conn.cursor()
    q = "SELECT date,competition,season,home_team,away_team FROM matches WHERE home_goals IS NULL"
    params = []

    if competition:
        q += " AND competition=?"
        params.append(competition)

    c.execute(q, params)
    rows = c.fetchall()

    out = []
    for date, comp, s, home, away in rows:
        pred = predict_match_poisson(home, away, models['atk'], models['defn'], models['avg_goals'])
        out.append({
            "date": date,
            "competition": comp,
            "home": home,
            "away": away,
            "pred": pred
        })

    return out
