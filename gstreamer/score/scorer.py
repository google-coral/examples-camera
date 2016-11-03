import pandas as pd
import csv
import datetime

class Scorer:
    csv_path = "score/scores.csv"

    def add(self, player_name:str="p1", noT:int=0):
        with open(self.csv_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([player_name, noT, datetime.date.today()])

    def get_all(self) -> pd.DataFrame:
        return pd.read_csv(self.csv_path)

    def get_score(self, start_date=datetime.date(2019, 1, 1), end_date=None) -> pd.DataFrame:
        end_date = end_date if end_date is not None else datetime.date.today()

        all_entries = self.get_all()
        all_entries = all_entries[(all_entries['date'] > start_date) & (all_entries['date'] < end_date)]

        total = all_entries.groupby("player")
