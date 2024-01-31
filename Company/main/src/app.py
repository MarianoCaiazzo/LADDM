# app.py

import time
import logging
import pandas as pd
import sys
sys.path.append("logs/")
from generate_logs import generate

def genera_log():
    df = generate(n=1)
    df = df.drop(columns=['label'])
    print(df.iloc[0].to_csv(index=False))
    df.to_csv("logs/company_log.csv",sep=";" , mode="a", header=False,  index=False)

# Configura il sistema di logging
# logging.basicConfig(filename='logs/company_log.txt', level=logging.INFO)
print("LOG APPLICATION STARTED")
# Esegui il servizio che genera log ogni tot secondi
input("Please click any keys to start")
while True:
    genera_log()
    time.sleep(10)  # intervallo di 10 secondi

