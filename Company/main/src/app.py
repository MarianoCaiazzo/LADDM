# app.py
import os
import time
import logging
import pandas as pd
import sys
sys.path.append("logs/")
from generate_logs import generate

def genera_log():
    file = open("logs/company_log.csv","a")
    data = generate(n=1)
    for d in data[0][:-1]:
        file.write(str(d)+",")
    file.write(str(data[0][-1])+"\n")
    file.close()

# Configura il sistema di logging
# logging.basicConfig(filename='logs/company_log.txt', level=logging.INFO)
print("LOG APPLICATION STARTING...")
# Esegui il servizio che genera log ogni tot secondi
input("Please click any keys to start")
columns = ["event","machine","timestamp","label"]
df = pd.DataFrame(generate(n=1))
if os.path.exists("logs/company_log.csv"):
     df.to_csv(
            "logs/company_log.csv",sep="," , mode="a", header=0 , index=False
        )
else:
    df.to_csv(
            "logs/company_log.csv",sep="," , mode="w", header=columns,
             index=False
        )

print("Generating logs...")
while True:
    genera_log()
    time.sleep(1)  # intervallo di 10 secondi

