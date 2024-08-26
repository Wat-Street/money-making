"""
This file contains functions to get data from our database tables.
"""

import datetime
from typing import List
import pandas as pd

class DataClient:
    def get_prices(
        ids: List[str], start_date: datetime.date, end_date: datetime.date, columns: List[str]
    ) -> pd.DataFrame:
        """_summary_

        Args:
            ids (List[str]): _description_
            start_date (datetime.date): _description_
            end_date (datetime.date): _description_
            columns (List[str]): _description_

        Returns:
            pd.DataFrame: _description_
        """
        pd.read_sql()
