import os
import yaml
import wandb
from ultralytics import YOLO
from codecarbon import EmissionsTracker
import logging
import sys
import time
import torch
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

logging.getLogger("codecarbon").setLevel(logging.CRITICAL)

# Directory di base
BASE_DIR = Path('/app')

# Directory per i risultati e i checkpoint
RESULTS_DIR = BASE_DIR / 'results'

# Assicurati che le directory esistano
os.makedirs(RESULTS_DIR, exist_ok=True)

def check_gradient_health(model):
    """Controlla la salute dei gradienti per prevenire NaN."""
    total_norm = 0
    param_count = 0
    nan_count = 0
    
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            if torch.isnan(param_norm) or torch.isinf(param_norm):
                nan_count += 1
                logging.warning(f"NaN/Inf gradient detected!")
                return False
            total_norm += param_norm.item() ** 2
            param_count += 1
    
    total_norm = total_norm ** (1. / 2)
    
    # Log gradient norm ogni tanto
    if param_count > 0:
        avg_norm = total_norm / param_count
        if avg_norm > 10.0:  # Soglia per gradienti troppo grandi
            logging.warning(f"Large gradient norm detected: {avg_norm:.4f}")
            return False
    
    return True

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


def setup_training_callbacks(model):
    """Imposta callback personalizzati per monitorare il training."""
    
    def on_train_epoch_end(trainer):
        """Callback chiamato alla fine di ogni epoca di training."""
        # Verifica la salute dei gradienti
        if hasattr(trainer, 'model') and trainer.model is not None:
            if not check_gradient_health(trainer.model):
                logging.error("Gradienti non salutari rilevati!")
                # Potresti decidere di fermare il training qui
                # trainer.stop_training = True
        
        # Log delle loss se disponibili
        if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
            losses = trainer.loss_items
            for i, loss in enumerate(losses):
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.error(f"NaN/Inf loss rilevata alla posizione {i}: {loss}")
    
    # Aggiungi callback al modello
    if hasattr(model, 'add_callback'):
        model.add_callback('on_train_epoch_end', on_train_epoch_end)

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
        logging.info(f"mAP50: {results.box.map50:.4f}, mAP50-95: {results.box.map:.4f}")
        return results

    except Exception as e:
        logging.error(f"Errore durante il test del modello: {e}")
        raise e


def train_model(model_config_path, dataset_config_path, weights_path=None):
    """Esegue l'addestramento del modello YOLOv12 con le configurazioni specificate."""
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
        tracker = EmissionsTracker(project_name=f"YOLOv12_{run_name}", output_dir=str(run_results_dir))
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
        resume_training = True
        if weights_path:
            logging.info(f"Caricamento del modello dai pesi pre-addestrati: {weights_path}")
            model = YOLO(weights_path)
            resume_training = False  # Inizia un nuovo training, non riprenderlo
        else:
            logging.info(f"Caricamento del modello dall'architettura: {model_config_path}")
            model_architecture_path = str(model_config_path)
            model = YOLO(model_architecture_path)

        # Imposta callback per il monitoraggio
        setup_training_callbacks(model)

        try:
            model.train(
                data=str(dataset_config_path),
                name=f"{run_name}",
                resume=resume_training,
                **model_config.get('training', {})
            )
        except Exception as e:
            logging.error(f"Errore durante l'addestramento di {run_name}: {e}")
            raise e

        # Termina il tracker di CO2 e ottieni i dati
        emissions = tracker.stop()
        
        # Registra i dati di CO2 su wandb
        if emissions is not None:
            wandb.run.summary["CO2_production_kg"] = emissions
            wandb.log({"CO2_production_kg": emissions})
            logging.info(f"Emissioni di CO2 per {run_name}: {emissions} kg")
            save_co2_data(emissions, run_results_dir)
        else:
            logging.warning("Impossibile ottenere dati sulle emissioni di CO2")

        try:
            # Test del modello
            test_model(model, run_name)

        except Exception as e:
            logging.error(f"Errore durante il test di {run_name}: {e}")
            raise e

    except Exception as e:
        logging.error(f"Errore durante l'addestramento di {run_name}: {e}")
        raise e
    finally:
        try:
            wandb.finish()
        except Exception as e:
            logging.error(f"Errore durante la chiusura di wandb: {e}")


def main():
    """Funzione principale per eseguire l'addestramento."""
    parser = argparse.ArgumentParser(description='Script per l\'addestramento di YOLOv12 con Ultralytics')
    parser.add_argument('--config', type=str, required=True, 
                        help='Percorso al file YAML di configurazione del modello')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Percorso al file YAML di configurazione del dataset')
    parser.add_argument('--weights', type=str, default=None,
                        help='Percorso opzionale ai pesi pre-addestrati (.pt)')

    args = parser.parse_args()

    model_config_path = Path(args.config)
    if not model_config_path.is_absolute():
        model_config_path = BASE_DIR / model_config_path

    dataset_config_path = Path(args.dataset)
    if not dataset_config_path.is_absolute():
        dataset_config_path = BASE_DIR / dataset_config_path

    weights_path = None
    if args.weights:
        weights_path = Path(args.weights)
        if not weights_path.is_absolute():
            weights_path = BASE_DIR / weights_path
        if not weights_path.exists():
            logging.error(f"File dei pesi non trovato: {weights_path}")
            sys.exit(1)

    if not model_config_path.exists():
        logging.error(f"File di configurazione del modello non trovato: {model_config_path}")
        sys.exit(1)

    if not dataset_config_path.exists():
        logging.error(f"File di configurazione del dataset non trovato: {dataset_config_path}")
        sys.exit(1)

    try:
        logging.info("="*80)
        logging.info("INIZIO TRAINING YOLOv12")
        logging.info("="*80)
        logging.info(f"Configurazione modello: {model_config_path}")
        logging.info(f"Configurazione dataset: {dataset_config_path}")
        if weights_path:
            logging.info(f"Pesi pre-addestrati: {weights_path}")
        logging.info("="*80)
        
        train_model(model_config_path, dataset_config_path, weights_path)
        
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