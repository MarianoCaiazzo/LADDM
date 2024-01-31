import pandas as pd
import sys
sys.path.append("logs/")
from generate_logs import generate

if __name__ == "__main__":
    print("Generate logs baseline for IA model")

    synthetic_dataset = generate(1000)
    columns = ["event","machine","timestamp","label"]
    synthetic_dataset.to_csv("logs/logFileBaseline.csv",sep=";",columns=columns,  index=False)

