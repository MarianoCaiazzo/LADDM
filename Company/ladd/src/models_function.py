import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from deepcase.preprocessing   import Preprocessor
from deepcase.interpreter import Interpreter
from deepcase.context_builder import ContextBuilder
from utily import normalize_tensor
from sklearn.preprocessing import LabelEncoder
from utily import PATTERN, regex_pattern
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_extraction.text import TfidfVectorizer



def plot_confusion_matrix(conf_matrix, class_names, path):
    plt.figure(figsize=(5, 5))
    sns.heatmap(
        conf_matrix, annot=True, fmt="d", cmap="Blues", linewidths=.5, 
        square=True, xticklabels=class_names, yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(path)


################################################################################

##                         DATASET SYTHETIC                                   ##

################################################################################

def create_synthetic_if_model(file_path, model_name, DATASET):
     # Gestisci il caso in cui il file non esiste
            print(f"Il file {model_name} non esiste. Verra Generato il Modello!")
            # elimina_timestamp(file_path, file_path_WO_TS="logs/logfileWithoutTS.csv")
            df = pd.read_csv(
                file_path, sep=",", names=["event","machine","timestamp","label"]
            )
            # df = df.drop(columns=['machine', 'timestamp'])
            # Esempio di dati di testo (sostituisci questo con i tuoi dati)
            # Utilizza TF-IDF per convertire il testo in una rappresentazione numerica
            train_df, test_df = train_test_split(
                df, test_size=0.2, random_state=42
            )
            vectorizer = TfidfVectorizer()
            X_train = vectorizer.fit_transform(train_df['event'])
            joblib.dump(vectorizer, f"models/if/{DATASET}/vectorized.pk1")
            # Contamination rappresenta la percentuale di anomalie previste nel dataset
            model_if = IsolationForest(contamination=0.1) 
            model_if.fit(X_train)
            joblib.dump(model_if, f"models/if/{DATASET}/{model_name}")
            evaluate_model_f(
                model_if, test_df, vectorized_model=True, DATASET=DATASET
            )
            return model_if


def create_synthetic_rf_model(file_path, model_name, DATASET):
     # Gestisci il caso in cui il file non esiste
            print(f"Il file {model_name} non esiste. Verra Generato il Modello!")
            df = pd.read_csv(
                file_path, 
                sep=",", 
                names=["event","machine","timestamp","label"]
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
            joblib.dump(rf_model, f'models/rf/{DATASET}/{model_name}')
            joblib.dump(encoder, f'models/rf/{DATASET}/encoder.joblib')
            train_df, test_df = train_test_split(
                df, test_size=0.2, random_state=42
            )
            evaluate_model_f(rf_model, test_df, DATASET, encoder_model=True)
            return rf_model


def create_synthetic_deepcase_model(file_path, model_name, DATASET):
    print(f"Il file {model_name} non esiste. Verra Generato il Modello!")
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
    
    context_builder.save(
        f'models/deepcase/{DATASET}/contextbuilder/ContextBuilder.save'
    )

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

    interpreter.save(f"models/deepcase/{DATASET}/interpreter/{model_name}")
    # evalute_model_DEEPCASE(interpreter, file_path)
    return interpreter


################################################################################

##                         DATASET HDFS                                       ##

################################################################################


    
def create_hdfs_if_model(file_path, model_name, DATASET="hdfs"):
    def elimina_alfa(lista):
        lista_valori = lista.split(",")
        values = []
        for val in lista_valori:
            temp = ""
            for v in val:
                if v.isdigit():
                    temp = temp+v
            values.append(int(temp))
        return sum(values)

    path = file_path.split("-")[0]
    df = pd.read_csv(path, sep=",")
    df_features = df[["Features", "Latency"]]
    df_features[['Features']] = df_features[['Features']].applymap(
                                                    lambda x: elimina_alfa(x)
    )
    labels_string = df[["Label"]]
    labels = labels_string.applymap(lambda x: 1 if x == "Success" else -1)
    X_train, X_test, y_train, y_test = train_test_split(
                            df_features, labels, test_size=0.2, random_state=42
    )

    model_if = IsolationForest(contamination=0.1)
    model_if.fit(X_train)
    joblib.dump(model_if, f"models/if/{DATASET}/{model_name}")
    anomalie_test_if = model_if.predict(X_test)
    matrice_confusione = confusion_matrix(y_test, anomalie_test_if)
    plot_confusion_matrix(
        matrice_confusione, 
        class_names=["Anomaly","Normal"], 
        path=f"logs/confusion_matrix/if/{DATASET}/cm_if.png"
    )
    # Visualizza la matrice di confusione
    print("Matrice di Confusione:")
    print(matrice_confusione)

    # Calcola e visualizza le metriche di classificazione
    print("\nReport di Classificazione:")
    print(classification_report(y_test, anomalie_test_if))


def create_hdfs_rf_model(file_path, model_name, DATASET="hdfs"):
    def elimina_alfa(lista):
        lista_valori = lista.split(",")
        values = []
        for val in lista_valori:
            temp = ""
            for v in val:
                if v.isdigit():
                    temp = temp+v
            values.append(int(temp))
        return sum(values)

    path = file_path.split("-")[0]
    df = pd.read_csv(path, sep=",")
    df_features = df[["Features", "Latency"]]
    labels_string = df[["Label"]]
    labels = labels_string.applymap(lambda x: 1 if x == "Success" else -1)
    df_features[['Features']] = df_features[['Features']].applymap(
                                                    lambda x: elimina_alfa(x)
    )
    X_train, X_test, y_train, y_test = train_test_split(
                            df_features, labels, test_size=0.2, random_state=42
    )

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_train)
    rf_accuracy = accuracy_score(y_train, rf_predictions)
    print("Accuracy Random Forest:", rf_accuracy)
    joblib.dump(rf_model, f"models/rf/{DATASET}/{model_name}")
    anomalie_test_rf = rf_model.predict(X_test)
    matrice_confusione = confusion_matrix(y_test, anomalie_test_rf)
    plot_confusion_matrix(
        matrice_confusione, 
        class_names=["Anomaly","Normal"], 
        path=f"logs/confusion_matrix/rf/{DATASET}/cm_rf.png"
    )

    # Visualizza la matrice di confusione
    print("Matrice di Confusione:")
    print(matrice_confusione)

    # Calcola e visualizza le metriche di classificazione
    print("\nReport di Classificazione:")
    print(classification_report(y_test, anomalie_test_rf))

def create_synthetic_deepcase_model(file_path, model_name, DATASET="hdfs"):
    path = "logs/" + file_path.split("-")[1]
    print(f"Il file {model_name} non esiste. Verra Generato il Modello!")
    preprocessor = Preprocessor(
    length  = 10,    # 10 events in context
    timeout = 86400, # Ignore events older than 1 day (60*60*24 = 86400 seconds)
    )

    context, events, labels, mapping = preprocessor.text(path, verbose=True)

    if labels is None:
        labels = np.full(events.shape[0], -1, dtype=int)

    # Split into train and test sets (20:80) by time - assuming events are ordered chronologically
    events_train_n  = events[:events.shape[0]//5 ]
    events_test   = events[ events.shape[0]//5:]

    context_train_n = context[:events.shape[0]//5 ]
    context_test  = context[ events.shape[0]//5:]

    label_train   = labels[:events.shape[0]//5 ]
    label_test    = labels[ events.shape[0]//5:]

    context_builder = ContextBuilder(
        input_size    = 100,   # Number of input features to expect
        output_size   = 100,   # Same as input size
        hidden_size   = 128,   # Number of nodes in hidden layer, in paper we set this to 128
        max_length    = 10,    # Length of the context, should be same as context in Preprocessor
    )

    context_builder.fit(
        X             = context_train_n,               # Context to train with
        y             = events_train_n.reshape(-1, 1), # Events to train with, note that these should be of shape=(n_events, 1)
        epochs        = 30,                          # Number of epochs to train with
        batch_size    = 128,                         # Number of samples in each training batch, in paper this was 128
        learning_rate = 0.01,                        # Learning rate to train with, in paper this was 0.01
        verbose       = True,                        # If True, prints progress
    )

    context_builder.save(
        f'models/deepcase/{DATASET}/contextbuilder/ContextBuilder.save'
    )

    ########################################################################
    #                  Get prediction from ContextBuilder                  #
    ########################################################################

    # Use context builder to predict confidence
    confidence, _ = context_builder.predict(
        X = context_test
    )

    # Get confidence of the next step, seq_len 0 (n_samples, seq_len, output_size)
    confidence = confidence[:, 0]
    # Get confidence from log confidence
    confidence = confidence.exp()
    # Get prediction as maximum confidence
    y_pred = confidence.argmax(dim=1)

    ########################################################################
    #                          Perform evaluation                          #
    ########################################################################

    # Get test and prediction as numpy array
    y_test = events_test.cpu().numpy()
    y_pred = y_pred     .cpu().numpy()

    # Print classification report
    print(classification_report(
        y_true = y_test,
        y_pred = y_pred,
        digits = 4,
    ))

    # Use context builder to predict confidence
    confidence, _ = context_builder.predict(
        X = context_test
    )

    # Get confidence of the next step, seq_len 0 (n_samples, seq_len, output_size)
    confidence = confidence[:, 0]
    # Get confidence from log confidence
    confidence = confidence.exp()
    # Get prediction as maximum confidence
    y_pred = confidence.argmax(dim=1)

    ########################################################################
    #                          Perform evaluation                          #
    ########################################################################

    # Get test and prediction as numpy array
    y_test = events_test.cpu().numpy()
    y_pred = y_pred     .cpu().numpy()

    # Print classification report
    print(classification_report(
        y_true = y_test,
        y_pred = y_pred,
        digits = 4,
    ))


################################################################################

##                         EVALUATE                                           ##

################################################################################

def evaluate_model_f(
    model, test_df, DATASET, vectorized_model = None, encoder_model=None
):
    path = None
    if vectorized_model:
        path = f"logs/confusion_matrix/if/{DATASET}/cm_if.png"
        vectorizer = joblib.load(f"models/if/{DATASET}/vectorized.pk1")
        nuovo_X_test = vectorizer.transform(test_df['event'])
    elif encoder_model:
        encoder = joblib.load(f'models/rf/{DATASET}/encoder.joblib')
        path = f"logs/confusion_matrix/rf/{DATASET}/cm_rf.png"
        # print(test_df['event'])
        nuovo_X_test = []
        for index, row in test_df.iterrows():
            action, n_bytes  = regex_pattern(row["event"], PATTERN)
            if action and n_bytes:
                action_n = int(encoder.transform([action])[0])
                value = list([action_n, (n_bytes)])
                nuovo_X_test.append(value)
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
    return True


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
    
    # Compute predicted scores
    predictions = interpreter.predict(
        X          = context_test,               # Context to predict
        y          = events_test.reshape(-1, 1), # Events to predict, note that these should be of shape=(n_events, 1)
        iterations = 100,                        # Number of iterations to use for attention query, in paper this was 100
        batch_size = 1024,                       # Batch size to use for attention query, used to limit CUDA memory usage
        verbose    = True,                       # If True, prints progress
    )
    
    return predictions
