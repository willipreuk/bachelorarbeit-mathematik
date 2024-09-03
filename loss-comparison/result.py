import pandas as pd
import numpy as np
from tabulate import tabulate
from main import functions

def print_results():
    for name, fun in functions.items():
        print(f"Function: {name}")
        csv_file_path =  f"results_{name}.csv"

        df = pd.read_csv(csv_file_path)

        df["rel_error_pred"] = np.abs((df["reference"] - df["predicted"]) / df["reference"])
        df["rel_error_cal"] = np.abs((df["reference"] - df["calculated"]) / df["reference"])
        df["rel_error_pred-cal"] = np.abs((df["calculated"] - df["predicted"]) / df["calculated"])

        df = df.groupby("function")
        df = df.agg({
            'loss': ['mean', 'std'],
            'mae': ['mean', 'std'],
            'rel_error_pred': ['mean','std'],
            'rel_error_cal': ['mean'],
            'rel_error_pred-cal': ['mean', 'std'],
        })
        df.sort_values(('rel_error_pred', 'mean'), inplace=True)

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        print(tabulate(df, headers='keys', tablefmt='github'))

        print("-" * 100)


if __name__ == '__main__':
    print_results()