# LADD
Log Anomaly Detection as Docker service
# For ladd and main
docker-compose build
docker-compose up -d

# Genera file log di partenza
docker exec -it main_main_1 python -u src/baseline_logs.py 
# Avvia la simulazione (genera 1 log al secondo)
docker exec -it main_main_1 python -u src/app.py

# Crea i modelli se non esistono
docker exec -e DATASET=syntetic -it ladd_ladd_1 python -u src/creation_model.py
# Avvia il controllo sul file di log
# MODE = 0 Tutti i modelli, = 1 DEEPCASE, = 2 IF, = 3 RF
docker exec -e NUM_RIGHE_DA_LEGGERE=1  MODE=0 -it ladd_ladd_1 python -u src/app.py
# docker exec -it ladd_ladd_1 pip install --default-timeout=5000 torch 