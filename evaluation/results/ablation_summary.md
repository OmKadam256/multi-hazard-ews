# Model Ablation & Evaluation Scorecard
Strict Spatial Split (Leave-Sub-Basin-Out) & Chronological Test Horizon

| Model Architecture | Flood PR-AUC | POD @ FAR=0.10 | CSI | Brier Score | Drought RMSE | Skill Score vs Persist |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1_Logistic_Regression** | 0.0000 | 0.0000 | 0.0000 | 0.0500 | 0.5447 | -15.6629 |
| **2_Random_Forest** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2584 | -2.7491 |
| **3_XGBoost** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2209 | -1.7405 |
| **4_Spatial_CNN_Only** | 0.0000 | 0.0000 | 0.0000 | 0.0511 | 0.5828 | -18.0694 |
| **5_Temporal_GRU_Only** | 0.0000 | 0.0000 | 0.0000 | 0.0442 | 0.6417 | -22.1218 |
| **6_MultiTask_Concatenation** | 0.0000 | 0.0000 | 0.0000 | 0.2809 | 2.0810 | -242.1742 |
| **7_MultiTask_GatedAttention_PROPOSED** | 0.0000 | 0.0000 | 0.0000 | 0.0520 | 0.3560 | -6.1171 |