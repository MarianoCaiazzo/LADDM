import time
import os
import joblib
import logging
from collections import deque
from creation_model import evalute_model_DEEPCASE

def leggi_file_condiviso(models, file_path, num_righe):
    try:
        with open(file_path, 'r') as file:
            righe = deque(file, maxlen=num_righe)
            for name_model, model in model.items():
                if name_model == "DEEPCASE":
                        normal_righe = predict_deepcase(model, righe)
                    for index, nr in enumerate(normal_righe):
                        if not nr:
                            logging.warning(f"{name_model} has detected as anomaky the following log: {riga[index]}")                        
            for riga in righe:
                print(riga, end='')
                    if name_model == "IF":
                        normal = predict_if(model, riga)
                    if normal is None:
                        logging.info(f"Error with analize the following log: {riga}")
                    if not normal:
                        logging.warning(f"{name_model} has detected as anomaky the following log: {riga}")
                

    except FileNotFoundError:
        print(f"Il file {file_path} non è stato trovato.")
    except Exception as e:
        print(f"Si è verificato un errore durante la lettura del file: {str(e)}")

if __name__ == "__main__":
    input("Please click any keys to start")
    models = load_models()
    while(True):
        percorso_file_condiviso = "logs/company_log.txt"
        num_righe_da_leggere = int(os.getenv("NUM_RIGHE_DA_LEGGERE", 10))
        leggi_file_condiviso(models, percorso_file_condiviso, num_righe_da_leggere-1)
        time.sleep(10)  # intervallo di 10 secondi
    
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
                        file_path = os.path.join(subfolder_path, filename)
                        # Verifica se l'elemento è un file
                        if os.path.isfile(file_path):
                            print(f"  - File trovato: {filename}")

                        if "deepcase" == subfolder:
                            interpreter = Interpreter.load(file_name)
                            models["DEEPCASE"] = interpreter
                        if "if" == subfolder:
                            model_if = joblib.load(filename) 
                            models["IF"] = model_if
        else:
            print(f"La cartella '{folder_path}' non esiste.")

        return models
    except Exception as e:
        print(f"Si è verificato un errore: {e}")


def predict_IF(model, value):
    nuovo_X_test = vectorizer.transform(value)
    # Effettua la previsione nel set di test
    anomalie_test_if = model_if.predict(nuovo_X_test)
    if anomalie_test_if == 0:
        logging.debug("Anomaly detected")
        return False
    if anomalie_test_if == 1:
        logging.debug("Normal Log")
        return True
    
    logging.debug(f"Error with Log {value} with predict value {anomalie_test_if}")
    return None

def predict_deepcase(model, values):       
    columns = ["event","machine","timestamp"]
    df = pd.DataFrame(values, columns=columns) 
    file_path = "logs/temp_logs_for_deep_case_predict.csv"
    df.to_csv(file_path, sep=";",columns=columns,  index=False)
    predicts = evalute_model_DEEPCASE(model, file_path)
    boolean_values = [True if x == -1 else False for x in predictions]
    return boolean_values