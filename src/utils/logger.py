from __future__ import annotations
from typing import List, Tuple
import atexit
import os

import pandas as pd

import utils.values as values



class EvolutionLogger:
    def __init__(self, columns: List[str]) -> None:
        self.name: str = ''
        self.round: int = 0
        # Logged data
        # We do not know beforehand how long the tuple will be
        self.data: List[Tuple] = []
        self.columns: List[str] = columns
        self._session_data: Tuple[str, int] = (self.name, self.round)
    
    def new_session(self, name: str, round: int) -> None:
        self.name = name
        self.round = round
        self._session_data = (self.name, self.round)
        print(f'Logging: {self._session_data}')
    
    def add_record(self, data: Tuple) -> None:
        self.data.append(self._session_data + data)
    
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.data, columns=['name', 'round'] + self.columns)
    
    def to_csv(self) -> None:
        self.to_dataframe().to_csv(os.path.join(values.RESULTS_DIR, 'evo_data.csv'), index=False)


# TODO: multiprocess, smarter global -> maybe local and add it to case callable?
glob_logger = EvolutionLogger(['generation', 'goal', 'fitness', 'strength', 'data_ratio', 'time_ratio'])

atexit.register(glob_logger.to_csv)