# Multimodal Ship Detection AI

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
- **Estendere** l'analisi usando YOLOv12 su input monocanale.
- **Ottimizzare** la pipeline per incrementare accuratezza ed efficienza del rilevamento.

---

## 🗂️ Struttura della Repository

```bash
.
├── configs/              # File di configurazione YOLO (hyp, data, model)
├── data/                 # Script per il preprocessing, chip extraction
├── models/               # Codice per YOLOv10slight e YOLOv12
├── notebooks/            # Notebook esplorativi (visualizzazioni, evaluation)
├── results/              # Risultati finali (confusion matrix, curve, immagini)
├── Figure/               # Immagini per slide/report
├── utils/                # Funzioni ausiliarie
├── requirements.txt      # Dipendenze Python
└── README.md             # Questo file
