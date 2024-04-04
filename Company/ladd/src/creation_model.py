import os
import numpy as np
import joblib
import traceback
from models_function import (
    create_synthetic_if_model, create_synthetic_rf_model, 
    create_synthetic_deepcase_model, create_hdfs_if_model,
    create_hdfs_rf_model, create_synthetic_deepcase_model
)
from deepcase.interpreter import Interpreter
from deepcase.context_builder import ContextBuilder

#MACRO
DATASET = os.getenv("DATASET","synthetic")


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
        if os.path.exists(f"models/rf/{DATASET}/{model_name}"):
            model = joblib.load(f"models/rf/{DATASET}/{model_name}")
            # Fai qualcosa con il modello, ad esempio utilizzalo per fare predizioni
            print(f"Modello {model_name} Esistente")
            return model
        else:
            if DATASET == "synthetic":
                return create_synthetic_rf_model(file_path, model_name, "synthetic" )
            elif DATASET == "hdfs":
                return create_hdfs_rf_model(file_path, model_name, "hdfs")
    except Exception as e:
        # Gestisci altre eccezioni se si verificano
        print(f"Si è verificato un errore: {e}")
        return None

def create_model_IF(file_path, model_name):
    try:
        if os.path.exists(f"models/if/{DATASET}/{model_name}"):
            model = joblib.load(f"models/if/{DATASET}/{model_name}")
            # Fai qualcosa con il modello, ad esempio utilizzalo per fare predizioni
            print(f"Modello {model_name} Esistente")
            return model
        else:
            if DATASET == "synthetic":
                return create_synthetic_if_model(file_path, model_name, "synthetic" )
            elif DATASET == "hdfs":
                return create_hdfs_if_model(file_path, model_name, "hdfs")
    except Exception as e:
        # Gestisci altre eccezioni se si verificano
        print(f"Si è verificato un errore: {e}")
        return None

def create_model_DEEPCASE(file_path, model_name):
    try:
        if os.path.exists(f"models/deepcase/{DATASET}/contextbuilder/ContextBuilder.save"):
            context_builder = ContextBuilder.load(
                f"models/deepcase/{DATASET}/contextbuilder/ContextBuilder.save"
            )
            if DATASET == "synthetic":
                if os.path.exists(f"models/deepcase/{DATASET}/interpreter/{model_name}") :
                    interpreter = Interpreter.load(
                        f"models/deepcase/{DATASET}/interpreter/{model_name}", context_builder
                    )
                    print(f"Interpreter Esistente")
                    return interpreter
                else:
                    create_synthetic_deepcase_model(file_path, model_name, "synthetic")
            else:
                print(f"Context Builder Esistente")
                return context_builder
        elif DATASET == "hdfs":
                create_synthetic_deepcase_model(file_path, model_name, "hdfs")
    except Exception as e:
        # Gestisci altre eccezioni se si verificano
        print(f"Si è verificato un errore: {e}")
        traceback.print_exc()

        return None


if __name__ == "__main__":
    print("Creation Model IA for LADD")
    models_name = {
        "IF":'isolation_forest_model.joblib',
        "DEEPCASE":'Interpreter.save',
        "RF": "random_forest.joblib"
    }
    path = "logFileBaseline.csv"
    if DATASET == "hdfs":
        path = "Event_traces.csv-hdfs_test_normal.txt"
    print(f"DATASET->{DATASET}")
    models = create_models(
        file_path=f"logs/{path}", models_name = models_name
    )
    # print(models)
    # df = pd.read_csv("logs/company_log.csv",sep=",", header=0)
    # # print(df.head())
    # train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    