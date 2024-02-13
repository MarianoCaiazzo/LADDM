import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import torch
import traceback
from utily import PATTERN, regex_pattern
from deepcase.preprocessing   import Preprocessor
from deepcase.interpreter import Interpreter
from deepcase.context_builder import ContextBuilder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
# from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# from datetime import datetime, timedelta
# import tensorflow as tf
# from tensorflow import keras
# from sklearn.preprocessing import StandardScaler

# def elimina_timestamp(file_path, file_path_WO_TS):
#   file_log_without_ts = open(file_path_WO_TS, "w")
#   with open(file_path, "r") as file_log_csv:
#     message_status = file_log_csv.readline()
#     file_log_without_ts.write(message_status)
#     data_read = file_log_csv.readlines()
#     for stringa in data_read:
#       s = stringa.split(" ")[:2]
#       file_log_without_ts.write(stringa.lstrip(s[0]).lstrip(" "+s[1]))
#       # print(stringa.lstrip(s[0]).lstrip(" "+s[1]))

def create_models(file_path, models_name):
    model_name_if = models_name.get("IF", None)
    model_if = create_model_IF(file_path, model_name_if)
    models = dict()
    models["IF"] = model_if
    model_name_rf = models_name.get("RF", None)
    model_rf = create_model_RF(file_path, model_name_rf)
    models["RF"] = model_rf
    model_name_deepcase = models_name.get("DEEPCASE", None)
    model_deepcase = create_model_DEEPCASE(file_path, model_name_deepcase)
    models["DEEPCASE"] = model_deepcase
    return models
    

def create_model_RF(file_path, model_name):
    try:
        # Prova ad aprire il file
        model = joblib.load(f"models/rf/{model_name}")
        # Fai qualcosa con il modello, ad esempio utilizzalo per fare predizioni
        print(f"Modello {model_name} Esistente")
        return model
    except FileNotFoundError:
        # Gestisci il caso in cui il file non esiste
        print(f"Il file {model_name} non esiste. Verra Generato il Modello!")
        # elimina_timestamp(file_path, file_path_WO_TS="logs/logfileWithoutTS.csv")
        df = pd.read_csv(
            file_path, sep=",", names=["event","machine","timestamp","label"]
        )
        
        dataset = ["Create", "Edit", "Login", "Logout"]
        encoder = LabelEncoder()
        valori_numerici = encoder.fit_transform(dataset)
        features = []
        labels = []
        for index, row in df.iterrows():
            if 'event' in row["event"]:
                continue
            action, n_bytes  = regex_pattern(row["event"], PATTERN)
            if action and n_bytes:
                action_n = int(encoder.transform([action])[0])
                value = list([action_n, int(n_bytes)])
                features.append(value)
                labels.append(row["label"])
                # print(action, n_bytes)
            else:
                print(f"Error with action: {action} and n_bytes: {n_bytes}")
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
            )
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_predictions = rf_model.predict(features)
        rf_accuracy = accuracy_score(labels, rf_predictions)
        print("Accuracy Random Forest:", rf_accuracy)
        joblib.dump(rf_model, f'models/rf/{model_name}')
        joblib.dump(encoder, f'models/rf/encoder.joblib')
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        evalute_model_f(rf_model, test_df, encoder_model=True)
        return rf_model
    except Exception as e:
        # Gestisci altre eccezioni se si verificano
        print(f"Si è verificato un errore: {e}")
        return None

def create_model_IF(file_path, model_name):
    try:
        # Prova ad aprire il file
        model = joblib.load(f"models/if/{model_name}")
        # Fai qualcosa con il modello, ad esempio utilizzalo per fare predizioni
        print(f"Modello {model_name} Esistente")
        return model
    except FileNotFoundError:
        # Gestisci il caso in cui il file non esiste
        print(f"Il file {model_name} non esiste. Verra Generato il Modello!")
        # elimina_timestamp(file_path, file_path_WO_TS="logs/logfileWithoutTS.csv")
        df = pd.read_csv(
            file_path, sep=",", names=["event","machine","timestamp","label"]
        )
        # df = df.drop(columns=['machine', 'timestamp'])
        # Esempio di dati di testo (sostituisci questo con i tuoi dati)
        # Utilizza TF-IDF per convertire il testo in una rappresentazione numerica
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        vectorizer = TfidfVectorizer()
        X_train = vectorizer.fit_transform(train_df['event'])
        joblib.dump(vectorizer, f"models/if/vectorized.pk1")
        # Contamination rappresenta la percentuale di anomalie previste nel dataset
        model_if = IsolationForest(contamination=0.1) 
        model_if.fit(X_train)
        joblib.dump(model_if, f"models/if/{model_name}")
        evalute_model_f(model_if, test_df, vectorized_model=True)
        return model_if
    except Exception as e:
        # Gestisci altre eccezioni se si verificano
        print(f"Si è verificato un errore: {e}")
        return None

def create_model_DEEPCASE(file_path, model_name):
    try:
        # Prova ad aprire il file
        context_builder = ContextBuilder.load(
            f"models/deepcase/contextbuilder/ContextBuilder.save"
        )
        interpreter = Interpreter.load(
            f"models/deepcase/interpreter/{model_name}", context_builder
        )
        # Fai qualcosa con il modello, ad esempio utilizzalo per fare predizioni
        print(f"Interpreter Esistente")
        return interpreter
    except FileNotFoundError:
        print(f"Il file {model_name} non esiste. Verra Generato il Modello!")
        traceback.print_exc() 
        preprocessor = Preprocessor(
        length  = 4,    # 10 events in context
        timeout = 86400, # Ignore events older than 1 day (60*60*24 = 86400 seconds)
        )

        context, events, labels, mapping = preprocessor.csv(file_path)
        
        if labels is None:
            labels = np.full(events.shape[0], -1, dtype=int)

        # Split into train and test sets (20:80) by time - assuming events are ordered chronologically
        events_train  = events[:events.shape[0]//5 ]
        # events_test   = events[ events.shape[0]//5:]

        context_train = context[:events.shape[0]//5 ]
        # context_test  = context[ events.shape[0]//5:]

        label_train   = labels[:events.shape[0]//5 ]
        # label_test    = labels[ events.shape[0]//5:]

        context_train_n = normalize_tensor(context_train)
        events_train_n = normalize_tensor(events_train)
        # context_test_n = normalize_tensor(context_test)
        # events_test_n = normalize_tensor(events_test)

        # Create ContextBuilder
        context_builder = ContextBuilder(
            input_size    = 100,   # Number of input features to expect
            output_size   = 100,   # Same as input size
            hidden_size   = 128,   # Number of nodes in hidden layer, in paper we set this to 128
            max_length    = 4,    # Length of the context, should be same as context in Preprocessor
        )

        context_builder.fit(
            X             = context_train_n,               # Context to train with
            y             = events_train_n.reshape(-1, 1), # Events to train with, note that these should be of shape=(n_events, 1)
            epochs        = 100,                          # Number of epochs to train with
            batch_size    = 128,                         # Number of samples in each training batch, in paper this was 128
            learning_rate = 0.01,                        # Learning rate to train with, in paper this was 0.01
            verbose       = True,                        # If True, prints progress
        )
        
        context_builder.save('models/deepcase/contextbuilder/ContextBuilder.save')

        # Create Interpreter
        interpreter = Interpreter(
            context_builder = context_builder, # ContextBuilder used to fit data
            features        = 100,             # Number of input features to expect, should be same as ContextBuilder
            eps             = 0.1,             # Epsilon value to use for DBSCAN clustering, in paper this was 0.1
            min_samples     = 5,               # Minimum number of samples to use for DBSCAN clustering, in paper this was 5
            threshold       = 0.2,             # Confidence threshold used for determining if attention from the ContextBuilder can be used, in paper this was 0.2
        )

        # Cluster samples with the interpreter
        clusters = interpreter.cluster(
            X          = context_train_n,               # Context to train with
            y          = events_train_n.reshape(-1, 1), # Events to train with, note that these should be of shape=(n_events, 1)
            iterations = 100,                         # Number of iterations to use for attention query, in paper this was 100
            batch_size = 1024,                        # Batch size to use for attention query, used to limit CUDA memory usage
            verbose    = True,                        # If True, prints progress
        )
        

        # Compute scores for each cluster based on individual labels per sequence
        scores = interpreter.score_clusters(
            scores   = label_train, # Labels used to compute score (either as loaded by Preprocessor, or put your own labels here)
            strategy = "max",        # Strategy to use for scoring (one of "max", "min", "avg")
            NO_SCORE = -1,           # Any sequence with this score will be ignored in the strategy.
                                    # If assigned a cluster, the sequence will inherit the cluster score.
                                    # If the sequence is not present in a cluster, it will receive a score of NO_SCORE.
        )

        # Assign scores to clusters in interpreter
        # Note that all sequences should be given a score and each sequence in the
        # same cluster should have the same score.
        interpreter.score(
            scores  = scores, # Scores to assign to sequences
            verbose = True,   # If True, prints progress
        )

        interpreter.save(f"models/deepcase/interpreter/{model_name}")
        # evalute_model_DEEPCASE(interpreter, file_path)
        return interpreter

    except Exception as e:
        # Gestisci altre eccezioni se si verificano
        print(f"Si è verificato un errore: {e}")
        traceback.print_exc()

        return None

def evalute_model_DEEPCASE(interpreter, file_path):
    preprocessor = Preprocessor(
        length  = 4,    # 10 events in context
        timeout = 86400, # Ignore events older than 1 day (60*60*24 = 86400 seconds)
        )

    context, events, labels, mapping = preprocessor.csv(file_path)
    
    if labels is None:
        labels = np.full(events.shape[0], -1, dtype=int)

    events_test   = events
    context_test  = context
    label_test    = labels
    # print(events)
    # print("________________________________________________")
    # print(context)
    # print("________________________________________________")
    # print(labels)
    # context_test_n = normalize_tensor(context_test)
    # events_test_n = normalize_tensor(events_test)
    # print(context_test_n)
    # print("________________________________________________")
    # print(events_test_n)
    # print("________________________________________________")


    # Compute predicted scores
    predictions = interpreter.predict(
        X          = context_test,               # Context to predict
        y          = events_test.reshape(-1, 1), # Events to predict, note that these should be of shape=(n_events, 1)
        iterations = 100,                        # Number of iterations to use for attention query, in paper this was 100
        batch_size = 1024,                       # Batch size to use for attention query, used to limit CUDA memory usage
        verbose    = True,                       # If True, prints progress
    )
    
    return predictions


def normalize_tensor(tensor_data):
  tensor_min = torch.min(tensor_data)
  tensor_max = torch.max(tensor_data)

  # Normalizza il Tensor nell'intervallo [0, 1]
  normalized_tensor = (tensor_data - tensor_min) / (tensor_max - tensor_min)

  # Puoi anche normalizzare in un intervallo specifico, ad esempio [0, 99]
  new_min = 0
  new_max = 99
  normalized_tensor_specific_range = \
   (tensor_data - tensor_min) / (tensor_max - tensor_min) \
        * (new_max - new_min) + new_min

  return normalized_tensor_specific_range.to(torch.long)


def plot_confusion_matrix(conf_matrix, class_names, path):
    plt.figure(figsize=(5, 5))
    sns.heatmap(
        conf_matrix, annot=True, fmt="d", cmap="Blues", linewidths=.5, 
        square=True, xticklabels=class_names, yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(path)

def evalute_model_f(model, test_df, vectorized_model = None, encoder_model=None):
    path = None
    if vectorized_model:
        path = "logs/confusion_matrix/if/cm_if.png"
        vectorizer = joblib.load("models/if/vectorized.pk1")
        nuovo_X_test = vectorizer.transform(test_df['event'])
    elif encoder_model:
        encoder = joblib.load(f'models/rf/encoder.joblib')
        path = "logs/confusion_matrix/rf/cm_rf.png"
        # print(test_df['event'])
        nuovo_X_test = []
        for index, row in test_df.iterrows():
            action, n_bytes  = regex_pattern(row["event"], PATTERN)
            if action and n_bytes:
                action_n = int(encoder.transform([action])[0])
                value = list([action_n, (n_bytes)])
                nuovo_X_test.append(value)
                # labels.append(row["label"])
                # print(action, n_bytes)
    else:
        print("Error, vectorized and encodere are None")
        return
    # Effettua la previsione nel set di test
    anomalie_test_if = model.predict(nuovo_X_test)
    anomalie_test_if = [int(i) for i in anomalie_test_if]
    labels = [int(i["label"]) for index, i in test_df.iterrows()]
    # etichette_effettive = [
    #     1 if etichetta == "Normal" else -1 for etichetta in test_df['Status']
    # ]
    matrice_confusione = confusion_matrix(labels, anomalie_test_if)
    plot_confusion_matrix(
        matrice_confusione, class_names=["Anomaly","Normal"], path=path
    )
    # Visualizza la matrice di confusione
    print("Matrice di Confusione:")
    print(matrice_confusione)

    # Calcola e visualizza le metriche di classificazione
    print("\nReport di Classificazione:")
    print(classification_report(labels, anomalie_test_if))

if __name__ == "__main__":
    print("Creation Model IA for LADD")
    models_name = {
        "IF":'isolation_forest_model.joblib',
        "DEEPCASE":'Interpreter.save',
        "RF": "random_forest.joblib"
    }
    models = create_models(
        file_path="logs/logFileBaseline.csv", models_name = models_name
    )
    # print(models)
    # df = pd.read_csv("logs/company_log.csv",sep=",", header=0)
    # # print(df.head())
    # train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    