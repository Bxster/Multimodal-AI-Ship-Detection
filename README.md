# Multimodal AI Ship Detection

> Sviluppo di un sistema AI multimodale per l'identificazione delle imbarcazioni con immagini satellitari  
> _Corso: Computer Vision & Deep Learning – Università Politecnica delle Marche_

---

## 🛰️ Descrizione del Progetto

Il progetto affronta la sfida della sorveglianza marittima integrando immagini satellitari (Sentinel-1 SAR e Sentinel-2 Ottiche) con tecniche avanzate di Deep Learning. L’obiettivo è sviluppare un sistema automatico per il rilevamento delle imbarcazioni, capace di superare i limiti dei sistemi tradizionali (AIS, rilevamento ottico/SAR isolato).

### 🔍 Contesto

- La sorveglianza marittima è critica per contrastare attività illegali (es. pesca IUU, traffici illeciti).
- I sistemi AIS risultano spesso disattivati ("dark vessels").
- Le immagini satellitari presentano problemi di interpretabilità (SAR) o visibilità (ottiche).

### 🎯 Obiettivi

- **Replicare** i risultati di [Galdelli et al. (2025)](https://www.sciencedirect.com/science/article/pii/S0167865525000649), con YOLOv10slight sul dataset HS3-S2.
- **Estendere** l'analisi usando YOLOv11 e YOLOv12 su input monocanale.
- **Ottimizzare** la pipeline per incrementare accuratezza ed efficienza del rilevamento, introducendo un modello Light della YOLOv12s.

---

## 🗂️ Struttura della Repository

```bash

├── yolov10/
│   ├── configs/                  # Configurazioni del modello YOLOv10 (es. YOLOv10sLight)
│   ├── dataset_configs/          # Configurazioni dei dataset (HRSID, D3 Grey)
│   └── train_yolov10.py          # Script di training per YOLOv10
│
├── yolov11/
│   ├── configs/                  # Configurazioni per YOLOv11 (YOLOv11s, YOLOv11m)
│   ├── dataset_configs/          # Configurazioni dei dataset (HRSID, D3 Grey)
│   └── train_yolov11.py          # Script di training per YOLOv11
│
├── yolov12/
│   ├── configs/                  # Configurazioni per YOLOv12
│   ├── dataset_configs/          # Configurazioni dei dataset (HRSID, D3 Grey)
│   ├── train_yolov12.py          # Script di training per YOLOv12
│   ├── train_yolov12_pre.py      # Script di training per YOLOv12 pre-addestrata
│   ├── best.pt                   # Pesi migliori del training con v12s
│   └── yolo12s.pt                # Modello pre-addestrato YOLOv12s
│
└── README.md                     # Documentazione del progetto
