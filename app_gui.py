
# app_gui.py
# Simple GUI to run ingestion, train, and show predictions.
import os, threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from mvp_predictor import DataStore, ingest_football_data_competition, ingest_openfootball_folder, build_models, predict_for_upcoming

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Soccer Predictor MVP')
        self.geometry('900x600')
        self.ds = DataStore()
        self._create_widgets()

    def _create_widgets(self):
        frm = ttk.Frame(self); frm.pack(fill='x', padx=8, pady=8)
        ttk.Label(frm, text='Competition code (football-data.org), e.g. PL, PD, SA').grid(row=0, column=0)
        self.comp_var = tk.StringVar(value='PL')
        ttk.Entry(frm, textvariable=self.comp_var, width=8).grid(row=1, column=0)
        ttk.Label(frm, text='Season (optional)').grid(row=0, column=1)
        self.season_var = tk.StringVar(value='2023')
        ttk.Entry(frm, textvariable=self.season_var, width=8).grid(row=1, column=1)
        ttk.Button(frm, text='Fetch from football-data.org', command=self.fetch_fd).grid(row=1, column=2, padx=6)
        ttk.Button(frm, text='Load openfootball folder', command=self.load_openfootball).grid(row=1, column=3, padx=6)
        ttk.Button(frm, text='Train models', command=self.train_models).grid(row=1, column=4, padx=6)

        mid = ttk.Frame(self); mid.pack(fill='both', expand=True, padx=8, pady=8)
        self.log = tk.Text(mid, height=20); self.log.pack(fill='both', expand=True)
        bottom = ttk.Frame(self); bottom.pack(fill='x', padx=8, pady=8)
        ttk.Button(bottom, text='Show Predictions', command=self.show_predictions).pack(side='left')

        self.models = None

    def log_print(self, s):
        self.log.insert('end', f'{s}\\n'); self.log.see('end')

    def fetch_fd(self):
        comp = self.comp_var.get().strip()
        season = self.season_var.get().strip() or None
        api = os.getenv('FOOTBALL_DATA_API_KEY')
        if not api:
            messagebox.showinfo('API key needed', 'Set environment variable FOOTBALL_DATA_API_KEY or use openfootball files.')
            return
        def job():
            try:
                n = ingest_football_data_competition(self.ds, comp, season, api_key=api)
                self.log_print(f'Fetched {n} matches for {comp}.')
            except Exception as e:
                self.log_print('Error: ' + str(e))
        threading.Thread(target=job, daemon=True).start()

    def load_openfootball(self):
        folder = filedialog.askdirectory(title='Select folder with openfootball JSON files')
        if not folder: return
        def job():
            n = ingest_openfootball_folder(self.ds, folder)
            self.log_print(f'Loaded {n} items from {folder}.')
        threading.Thread(target=job, daemon=True).start()

    def train_models(self):
        def job():
            self.log_print('Training models...')
            self.models = build_models(self.ds)
            self.log_print('Models built.')
        threading.Thread(target=job, daemon=True).start()

    def show_predictions(self):
        if not self.models:
            messagebox.showinfo('No models', 'Train models first.')
            return
        preds = predict_for_upcoming(self.ds, self.models)
        if not preds:
            messagebox.showinfo('No upcoming matches', 'There are no matches without results in the DB. Import fixtures first.')
            return
        win = tk.Toplevel(self); win.title('Predictions'); win.geometry('700x500')
        tree = ttk.Treeview(win, columns=('date','comp','home','away','prob_win','prob_draw','prob_loss','top_score'), show='headings')
        for c in ('date','comp','home','away','prob_win','prob_draw','prob_loss','top_score'):
            tree.heading(c, text=c); tree.column(c, width=90)
        tree.pack(fill='both', expand=True)
        for p in preds:
            top = p['pred']['top_scores'][0][0]
            tree.insert('', 'end', values=(p['date'], p['competition'], p['home'], p['away'],
                                          f\"{p['pred']['p_win']:.2f}\", f\"{p['pred']['p_draw']:.2f}\", f\"{p['pred']['p_loss']:.2f}\", f\"{top[0]}-{top[1]}\"))

if __name__ == '__main__':
    app = App(); app.mainloop()
