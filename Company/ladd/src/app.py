import time
import os
import joblib
import logging
import traceback
import pandas as pd
from utily import PATTERN, regex_pattern
from collections import deque
from creation_model import evalute_model_DEEPCASE
from deepcase.interpreter import Interpreter
from deepcase.context_builder import ContextBuilder
from sklearn.feature_extraction.text import TfidfVectorizer


def predict_deepcase(model, values):       
    columns = ["event","machine","timestamp","label"]
    value_list = []
    for val in values:
        lista = []
        for v in val.split(","):
            lista.append(v.replace("\n", "") )
        value_list.append(lista)
    # value_list = [val.split(",") for val in values]
    # print(value_list)
    df = pd.DataFrame(value_list, columns=columns) 
    file_path = "logs/temp_logs_for_deep_case_predict.csv"
    df.to_csv(file_path, sep=",", index=False)
    predicts = evalute_model_DEEPCASE(model, file_path)
    boolean_values = [True if x == -1 else False for x in predicts]
    return boolean_values

def predict_if(model, value):
    vectorizer = joblib.load("models/if/vectorized.pk1")
    event = value.split(",")[0]
    nuovo_X_test = vectorizer.transform([event])
    # Effettua la previsione nel set di test
    anomalie_test_if = model.predict(nuovo_X_test)
    print(event)
    # print(anomalie_test_if)
    if anomalie_test_if == -1:
        logging.debug("Anomaly detected")
        return False
    if anomalie_test_if == 1:
        logging.debug("Normal Log")
        return True
    
    logging.debug(
        f"Error with Log {value} with predict value {anomalie_test_if}"
    )
    return None

def predict_rf(model, value):
    encoder = joblib.load(f'models/rf/encoder.joblib')
    event = value.split(",")[0]
    # Effettua la previsione nel set di test
    action, n_bytes  = regex_pattern(event, PATTERN)
    # print(action, n_bytes)
    if action and n_bytes:
        action_n = int(encoder.transform([action])[0])
        nuovo_X_test = [[action_n, int(n_bytes)]]
        anomalie_test_rf = model.predict(nuovo_X_test)
        anomalie_test_rf = int(anomalie_test_rf[0])
        # print(event)
        # print(anomalie_test_rf)
        # print(anomalie_test_if)
        if anomalie_test_rf == -1:
            logging.debug("Anomaly detected")
            return False
        if anomalie_test_rf == 1:
            logging.debug("Normal Log")
            return True
        
        logging.debug(
            f"Error with Log {value} with predict value {anomalie_test_rf}"
        )
    return None


def load_models():
    folder_path = 'models'
    models = dict()
    try:
        # Verifica se la cartella esiste
        if os.path.exists(folder_path):
            # Scorri tutte le cartelle nella cartella principale
            for subfolder in os.listdir(folder_path):
                subfolder_path = os.path.join(folder_path, subfolder)
                # Verifica se l'elemento è una cartella
                if os.path.isdir(subfolder_path):
                    print(f"Cartella trovata: {subfolder}")
                    # Scorri tutti i file nella sottocartella
                    for filename in os.listdir(subfolder_path):
                        if ".pk1" in filename:
                            continue
                        file_path = os.path.join(subfolder_path, filename)
                        # Verifica se l'elemento è un file
                        if os.path.isfile(file_path):
                            print(f"  - File trovato: {filename}")
                        if "deepcase" == subfolder:
                            deepcase_components = []
                            for component in os.listdir(
                                f"{subfolder_path}/interpreter"
                            ):
                                deepcase_components.append(
                                    f"{subfolder_path}/interpreter/{component}"
                                )
                            for component in os.listdir(
                                f"{subfolder_path}/contextbuilder"
                            ):
                                deepcase_components.append(
                                    f"{subfolder_path}/contextbuilder/{component}"
                                )
                            context_builder = ContextBuilder.load(
                                deepcase_components[1]
                            )
                            interpreter = Interpreter.load(
                                deepcase_components[0], context_builder
                            )
                            models["DEEPCASE"] = interpreter
                        if "if" == subfolder:
                            model_if = joblib.load(file_path) 
                            models["IF"] = model_if
                        if "rf" == subfolder:
                            model_if = joblib.load(file_path) 
                            models["RF"] = model_if
        else:
            print(f"La cartella '{folder_path}' non esiste.")

        return models
    except Exception as e:
        print(f"Si è verificato un errore: {e}")
        traceback.print_exc()


def leggi_file_condiviso(models, file_path, num_righe):
    try:
        file_error =  open("logs/errors/anomalies.csv", "a")
    except: 
        file_error =  open("logs/errors/anomalies.csv", "w")
    try:
        df = pd.read_csv(
            "logs/errors/anomalies.csv", 
            sep=",",names=["event","machine","timestamp","label","model_name"]
        )
        with open(file_path, 'r') as file:
            righe = list(deque(file, maxlen=num_righe))
            for name_model, model in models.items():
                if name_model == "DEEPCASE":
                    continue
                    normal_righe = predict_deepcase(model, righe)
                    print(f"Boolean Values {normal_righe}")
                    for index, nr in enumerate(normal_righe):
                        if not nr:
                            print(
                                name_model+
                                " has detected as anomaky the following log: "+
                                righe[index]
                            )
                            print_file = \
                                righe[index].replace("\n","")+","+name_model+"\n"
                            timestamp = int(righe[index].split(",")[2])
                            if not df[df["timestamp"] == timestamp]["event"].any(): 
                                file_error.write(print_file)                         
                # print(riga, end='')
                if name_model == "IF":
                    continue
                    if "event" in riga:
                        continue
                    for riga in righe:
                        normal = predict_if(model, riga)
                        # print(normal)
                        if normal is None:
                            logging.info(
                                f"Error with analize the following log: {riga}"
                            )
                        if not normal:
                            print_file = riga.replace("\n", "")+","+name_model+"\n"
                            print(
                                name_model+ 
                                " has detected as anomaky the following log: "+
                                riga
                            )
                            timestamp = int(riga.split(",")[2])
                            # print(df.head())
                            # print("_____________________________________________________")
                            # print(df[df["timestamp"] == timestamp]["event"])
                            # print(df[df["timestamp"] == timestamp]["event"].any())
                            if not df[df["timestamp"] == timestamp]["event"].any(): 
                                #Se non è stato gia inserita la riga con lo stesso timestamp
                                file_error.write(print_file)
                if name_model == "RF":
                    for riga in righe:
                        if "event" in riga:
                            continue
                        normal = predict_rf(model, riga)
                        # print(normal)
                        if normal is None:
                            logging.info(
                                f"Error with analize the following log: {riga}"
                            )
                        if not normal:
                            print_file = riga.replace("\n", "")+","+name_model+"\n"
                            print(
                                name_model+ 
                                " has detected as anomaky the following log: "+
                                riga
                            )
                            timestamp = int(riga.split(",")[2])
                            # print(df.head())
                            # print("_____________________________________________________")
                            # print(df[df["timestamp"] == timestamp]["event"])
                            # print(df[df["timestamp"] == timestamp]["event"].any())
                            if not df[df["timestamp"] == timestamp]["event"].any(): 
                                #Se non è stato gia inserita la riga con lo stesso timestamp
                                file_error.write(print_file)
    except Exception as e:
        print(f"Si è verificato un errore: {str(e)}")
        traceback.print_exc()
    finally:
        file_error.close()



if __name__ == "__main__":
    input("Please click any keys to start")
    models = load_models()
    while(True):
        percorso_file_condiviso = "logs/company_log.csv"
        num_righe_da_leggere = int(os.getenv("NUM_RIGHE_DA_LEGGERE", 10))
        # print(num_righe_da_leggere)
        leggi_file_condiviso(
            models, percorso_file_condiviso, num_righe_da_leggere
        )
        time.sleep(1)  # intervallo di 10 secondi
    


