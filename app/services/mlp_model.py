"""
MLP + MC Dropout pour prédiction de rendement tomate
Remplace LightGBM + kNN par un modèle neuronal unifié
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple
import joblib
from datetime import datetime, timedelta


class MCDropoutMLP(nn.Module):
    """
    MLP avec Dropout à chaque couche cachée.
    MC Dropout : garder dropout.train() même à l'inférence
    → N passes forward → moyenne ± std = prédiction ± incertitude
    """
    def __init__(self, n_input: int, hidden: list[int] = [256, 128, 64], dropout_rate: float = 0.3):
        super().__init__()
        layers = []
        prev = n_input
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def mc_predict(self, x_tensor: torch.Tensor, n_passes: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        n_passes forward avec dropout actif → distribution de prédictions.
        Retourne (moyenne, std) en espace log.
        """
        # Si batch_size=1, BatchNorm peut planter. On duplique l'input temporairement
        if x_tensor.size(0) == 1:
            x_tensor = torch.cat([x_tensor, x_tensor], dim=0)  # [1, features] → [2, features]
            take_first_only = True
        else:
            take_first_only = False
            
        self.train()   # dropout actif même en inférence ← clé du MC Dropout
        with torch.no_grad():
            preds = torch.stack([self(x_tensor) for _ in range(n_passes)], dim=0)
        self.eval()
        
        mean_pred = preds.mean(dim=0)
        std_pred = preds.std(dim=0)
        
        # Prendre seulement le premier échantillon si on a dupliqué
        if take_first_only:
            mean_pred = mean_pred[0]
            std_pred = std_pred[0]
            
        return mean_pred, std_pred


class RendementPredictor:
    """
    Prédicteur de rendement unifié utilisant MLP + MC Dropout
    """
    
    def __init__(self, model_path: str = "mlp_model.pt"):
        self.model_path = Path(model_path)
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.is_loaded = False
        
    def load_model(self):
        """Charge le modèle et les métadonnées depuis le fichier PyTorch"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modèle non trouvé : {self.model_path}")
            
        checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
        
        # Reconstruit l'architecture
        n_features = len(checkpoint['feat_cols'])
        self.model = MCDropoutMLP(n_features, hidden=[256, 128, 64], dropout_rate=0.3)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        
        # Scaler et features
        self.scaler_mean = checkpoint['scaler_mean']
        self.scaler_std = checkpoint['scaler_std']
        self.feature_cols = checkpoint['feat_cols']
        self.is_loaded = True
        
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalise les features avec le scaler sauvegardé"""
        return (features - self.scaler_mean) / self.scaler_std
    
    def _extract_features_from_api(self, bands: dict, weather: dict, 
                                   date_plantation: str, date_prediction: str) -> dict:
        """
        Extrait et calcule les features à partir des données API
        SIMULATION pour la démo — dans la vraie vie, ces features viendraient 
        du pipeline complet EZZAYRA
        """
        # Calculs temporels
        plantation = datetime.strptime(date_plantation, "%Y-%m-%d")
        prediction = datetime.strptime(date_prediction, "%Y-%m-%d")
        days_since_planting = (prediction - plantation).days
        
        # Features simulées basées sur les inputs API
        ndvi_series = bands.get('ndvi_series', [0.3, 0.5, 0.7, 0.6])
        precip_series = weather.get('precipitation_mm', [5, 12, 8, 3, 0])
        temp_series = weather.get('temperature_2m', [22, 25, 28, 30, 27])
        
        # Agrégats par phase phénologique simulée
        features = {
            'ndvi_mean_s2': np.mean(ndvi_series[:2]) if len(ndvi_series) > 1 else 0.4,
            'ndvi_max_s2': np.max(ndvi_series[:2]) if len(ndvi_series) > 1 else 0.5,
            'ndvi_std_s2': np.std(ndvi_series[:2]) if len(ndvi_series) > 1 else 0.1,
            'et0_mean_s2': 4.2,  # valeur typique Tunisie
            'temp_min_s2': np.min(temp_series[:3]) if len(temp_series) > 2 else 20,
            'precip_cum_s3': np.sum(precip_series[2:5]) if len(precip_series) > 4 else 15,
            'stress_hydrique_s3': max(0, 30 - np.sum(precip_series[2:5])),
            'gdd_s2': days_since_planting * 0.8,  # approximation GDD
            'bdod': 1.4,  # densité sol typique
            'clay': 25.0,  # % argile typique
            'sand': 45.0,  # % sable typique
            'area_polygon': 2.5,  # ha approximatif
        }
        
        # Complète avec des zéros pour atteindre le bon nombre de features
        # En production, toutes les 178 features seraient calculées
        for i in range(len(features), 178):
            features[f'feature_{i}'] = 0.0
            
        return features
    
    def predict(self, bands: dict, weather: dict, date_plantation: str, 
                date_prediction: str) -> dict:
        """
        Prédiction complète avec intervalle de confiance et flags d'incertitude
        
        Returns:
            dict avec tonnage_predit_t, intervalle_confiance_95, 
            incertitude_niveau, top_features_shap (simulé)
        """
        if not self.is_loaded:
            self.load_model()
            
        # Extraction des features
        features_dict = self._extract_features_from_api(
            bands, weather, date_plantation, date_prediction
        )
        
        # Conversion en array ordonné selon feature_cols
        if len(features_dict) < len(self.feature_cols):
            # Complète les features manquantes avec des zéros
            for col in self.feature_cols:
                if col not in features_dict:
                    features_dict[col] = 0.0
        
        feature_array = np.array([features_dict.get(col, 0.0) for col in self.feature_cols])
        feature_array = feature_array.reshape(1, -1).astype(np.float32)
        
        # Normalisation
        features_norm = self._normalize_features(feature_array)
        x_tensor = torch.tensor(features_norm, dtype=torch.float32)
        
        # Prédiction MC Dropout
        mean_log, std_log = self.model.mc_predict(x_tensor, n_passes=50)
        
        # Conversion en espace linéaire
        tonnage_pred = np.exp(mean_log.item())
        
        # Intervalle de confiance 95%
        ci_low = np.exp(mean_log.item() - 1.96 * std_log.item())
        ci_high = np.exp(mean_log.item() + 1.96 * std_log.item())
        
        # Flag d'incertitude (seuil calibré sur validation)
        uncertainty_level = "NORMALE"
        if std_log.item() > 0.45:  # seuil empirique
            uncertainty_level = "HAUTE"
        elif tonnage_pred < 30 or tonnage_pred > 120:
            uncertainty_level = "HAUTE"  # prédictions extrêmes
            
        # SHAP simulé (en production, calculé avec le vrai explainer)
        top_shap = [
            {"feature": "ndvi_max_s2", "impact": 8.4},
            {"feature": "stress_hydrique_s3", "impact": -3.1}, 
            {"feature": "temp_min_s2", "impact": 2.7},
            {"feature": "precip_cum_s3", "impact": -1.8},
            {"feature": "bdod", "impact": 1.2}
        ]
        
        # Date de récolte estimée (plantation + ~120 jours)
        plantation_date = datetime.strptime(date_plantation, "%Y-%m-%d")
        harvest_date = plantation_date + timedelta(days=120)
        
        return {
            "tonnage_predit_t": round(tonnage_pred, 1),
            "intervalle_confiance_95": [round(ci_low, 1), round(ci_high, 1)],
            "incertitude_niveau": uncertainty_level,
            "incertitude_sigma_log": round(std_log.item(), 3),
            "top_features_shap": top_shap,
            "date_recolte_estimee": harvest_date.strftime("%Y-%m-%d")
        }


# Instance globale pour éviter de recharger le modèle
_predictor = None

def get_predictor() -> RendementPredictor:
    """Singleton pour le prédicteur"""
    global _predictor
    if _predictor is None:
        _predictor = RendementPredictor()
    return _predictor