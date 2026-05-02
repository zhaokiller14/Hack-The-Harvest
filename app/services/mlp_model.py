"""
MLP + MC Dropout pour prédiction de rendement tomate
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple
from datetime import datetime, timedelta

from app.services.feature_builder import build_features


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
    
    def __init__(self, model_path: str = str(Path(__file__).resolve().parent.parent.parent / "models" / "mlp_model.pt")):
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
    
    def predict(self, bands: dict, weather: dict, date_plantation: str,
                date_prediction: str, soil: dict | None = None,
                area_ha: float = 2.5) -> dict:
        """
        Prédiction complète avec intervalle de confiance et flags d'incertitude
        
        Returns:
            dict avec tonnage_predit_t, intervalle_confiance_95, 
            incertitude_niveau, top_features_shap (simulé)
        """
        if not self.is_loaded:
            self.load_model()

        # Build full feature dict from all data sources
        features_dict = build_features(
            weather=weather,
            soil=soil or {},
            bands=bands,
            area_ha=area_ha,
        )

        feature_array = np.array([features_dict.get(col, 0.0) for col in self.feature_cols])
        feature_array = feature_array.reshape(1, -1).astype(np.float32)
        
        # Normalisation
        features_norm = self._normalize_features(feature_array)
        x_tensor = torch.tensor(features_norm, dtype=torch.float32)
        
        # Prédiction MC Dropout
        mean_log, std_log = self.model.mc_predict(x_tensor, n_passes=50)
        
        # Conversion en espace linéaire
        tonnage_pred = np.exp(mean_log.item())
        
        # Intervalle de confiance 95% — clamped to physically observed range (11–165 t/ha)
        ci_low  = max(10.0, np.exp(mean_log.item() - 1.96 * std_log.item()))
        ci_high = min(200.0, np.exp(mean_log.item() + 1.96 * std_log.item()))
        
        # Flag d'incertitude (seuil calibré sur validation)
        uncertainty_level = "NORMALE"
        if std_log.item() > 0.45:  # seuil empirique
            uncertainty_level = "HAUTE"
        elif tonnage_pred < 30 or tonnage_pred > 120:
            uncertainty_level = "HAUTE"  # prédictions extrêmes
            
        # Approximate SHAP via feature sensitivity (gradient × feature value)
        x_tensor.requires_grad_(True)
        pred_for_grad = self.model(x_tensor)
        pred_for_grad.backward()
        gradients = x_tensor.grad.detach().numpy().flatten()
        x_tensor.requires_grad_(False)
        sensitivity = gradients * features_norm.flatten()
        top_idx = abs(sensitivity).argsort()[::-1][:5]
        top_shap = [
            {"feature": self.feature_cols[i], "impact": round(float(sensitivity[i]), 3)}
            for i in top_idx
        ]
        
        # Date de récolte estimée (plantation + ~120 jours)
        plantation_date = datetime.strptime(date_plantation, "%Y-%m-%d")
        harvest_date = plantation_date + timedelta(days=120)
        
        return {
            "tonnage_predit_t": round(float(tonnage_pred), 1),
            "intervalle_confiance_95": [round(float(ci_low), 1), round(float(ci_high), 1)],
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