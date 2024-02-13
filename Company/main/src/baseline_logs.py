import pandas as pd
import sys
sys.path.append("logs/")
from generate_logs import generate

if __name__ == "__main__":
    print("Generate logs baseline for IA model")

    synthetic_dataset =  pd.DataFrame(
        generate(1000),columns=["event","machine","timestamp","label"]
    )
    synthetic_dataset = synthetic_dataset.drop_duplicates(subset="event")
    # columns = ["event","machine","timestamp","label"]
    synthetic_dataset.to_csv(
        "logs/logFileBaseline.csv",sep=",",
        index=False, header=["event","machine","timestamp","label"]
    )

