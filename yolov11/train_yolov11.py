import os
import yaml
import wandb
from ultralytics import YOLO
from codecarbon import EmissionsTracker
import logging
import sys
import time
from pathlib import Path
import argparse

# Configurazione del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Directory di base
BASE_DIR = Path('/app')

# Directory per i risultati e i checkpoint
RESULTS_DIR = BASE_DIR / 'results'

# Assicurati che le directory esistano
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_yaml(file_path):
    """Carica un singolo file YAML."""
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    logging.info(f"Configurazione caricata: {file_path}")
    return data


def save_co2_data(co2_data, run_results_dir):
    """Salva i dati di produzione di CO2 in un file locale."""
    co2_file = run_results_dir / 'co2_production.txt'
    with open(co2_file, 'a') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - CO2: {co2_data} kg\n")
    logging.info(f"CO2 salvato in {co2_file}")


def test_model(model, run_name):
    """Esegue il test del modello sul dataset di test."""
    logging.info("Inizio del test del modello...")
    try:
        # Esegui il test utilizzando il set di test definito nel dataset YAML
        results = model.val(
            split="test",
            save_txt=True,
            save_json=True,
            save_conf=True,
            name=f"{run_name}_test",
            verbose=True
        )
        
        logging.info(f"Test completato per {run_name}")
        return results

    except Exception as e:
        logging.error(f"Errore durante il test del modello: {e}")
        raise e


def validate_model(model, run_name):
    """Esegue la validazione del modello sul dataset di validazione."""
    logging.info("Inizio della validazione del modello...")
    try:
        # Esegui la validazione utilizzando il set di validazione
        results = model.val(
            split="val",
            save_txt=True,
            save_json=True,
            save_conf=True,
            name=f"{run_name}_val",
            verbose=True
        )
        
        logging.info(f"Validazione completata per {run_name}")
        return results

    except Exception as e:
        logging.error(f"Errore durante la validazione del modello: {e}")
        raise e


def train_model(model_config_path, dataset_config_path):
    """Esegue l'addestramento del modello YOLOv10 con le configurazioni specificate."""
    try:
        model_config = load_yaml(model_config_path)
        dataset_config = load_yaml(dataset_config_path)

        # Configura i nomi per il salvataggio dei risultati
        config_name = Path(model_config_path).stem
        dataset_name = Path(dataset_config_path).stem
        run_name = f"{config_name}_{dataset_name}"
        run_results_dir = RESULTS_DIR / run_name
        os.makedirs(run_results_dir, exist_ok=True)

        # Inizializza il tracker di CO2
        tracker = EmissionsTracker(project_name=f"YOLOv11_{run_name}", output_dir=str(run_results_dir))
        tracker.start()

        def check_for_test_tag(name: str):
            return "Test" in name or "test" in name

        # Verifica e imposta i tag
        if check_for_test_tag(run_name):
            wandb.init(
                project="results",
                name=f"{run_name}",
                tags=["test"]
            )
        else:
            wandb.init(
                project="results",
                name=f"{run_name}",
            )

        # Carica il modello con Ultralytics
        model_architecture_path = str(model_config_path)
        model = YOLO(model_architecture_path)

        try:
            model.train(
                data=str(dataset_config_path),
                name=f"{run_name}",
                resume=True,
                **model_config.get('training', {})
            )
        except Exception as e:
            logging.error(f"Errore durante l'addestramento di {run_name}: {e}")
            raise e

        # Termina il tracker di CO2 e ottieni i dati
        emissions = tracker.stop()
        
        # Registra i dati di CO2 su wandb
        wandb.run.summary["CO2_production_kg"] = emissions
        wandb.log({"CO2_production_kg": emissions})
        logging.info(f"Emissioni di CO2 per {run_name}: {emissions} kg")

        # Salva i dati di CO2 in un file locale
        save_co2_data(emissions, run_results_dir)

        try:
            # Test del modello
            test_model(model, run_name)

        except Exception as e:
            logging.error(f"Errore durante il test di {run_name}: {e}")
            raise e

        finally:
            try:
                wandb.finish()
            except Exception as e:
                logging.error(f"Errore durante la chiusura di wandb: {e}")

    except Exception as e:
        logging.error(f"Errore durante l'addestramento di {run_name}: {e}")
        try:
            wandb.finish()
        except:
            pass
        raise e


def main():
    """Funzione principale per eseguire l'addestramento."""
    parser = argparse.ArgumentParser(description='Script per l\'addestramento di YOLOv11 con Ultralytics')
    parser.add_argument('--config', type=str, required=True, 
                        help='Percorso al file YAML di configurazione del modello')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Percorso al file YAML di configurazione del dataset')

    args = parser.parse_args()

    model_config_path = Path(args.config)
    if not model_config_path.is_absolute():
        model_config_path = BASE_DIR / model_config_path

    dataset_config_path = Path(args.dataset)
    if not dataset_config_path.is_absolute():
        dataset_config_path = BASE_DIR / dataset_config_path

    if not model_config_path.exists():
        logging.error(f"File di configurazione del modello non trovato: {model_config_path}")
        sys.exit(1)

    if not dataset_config_path.exists():
        logging.error(f"File di configurazione del dataset non trovato: {dataset_config_path}")
        sys.exit(1)

    try:
        logging.info("="*80)
        logging.info("INIZIO TRAINING YOLOv11")
        logging.info("="*80)
        logging.info(f"Configurazione modello: {model_config_path}")
        logging.info(f"Configurazione dataset: {dataset_config_path}")
        logging.info("="*80)
        
        train_model(model_config_path, dataset_config_path)
        
        logging.info("="*80)
        logging.info(f"TRAINING COMPLETATO CON SUCCESSO: {model_config_path}")
        logging.info("="*80)
        
    except Exception as e:
        logging.error("="*80)
        logging.error(f"TRAINING INTERROTTO: {model_config_path}")
        logging.error(f"Errore: {e}")
        logging.error("="*80)
        sys.exit(1)


if __name__ == "__main__":
    main()