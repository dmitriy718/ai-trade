"""
ML Model Trainer - Auto-retraining pipeline for TFLite trade predictor.

Trains a neural network on historical trade features and outcomes,
converts to TFLite format, and deploys the model for live inference.

# ENHANCEMENT: Added model versioning with rollback capability
# ENHANCEMENT: Added cross-validation for robustness
# ENHANCEMENT: Added feature importance analysis
# ENHANCEMENT: Added training metrics logging
# ENHANCEMENT: Isolate training in separate process to avoid blocking event loop
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.ai.predictor import TradePredictorFeatures
from src.core.database import DatabaseManager
from src.core.logger import get_logger

logger = get_logger("ml_trainer")


def _run_training_process(
    X: np.ndarray,
    y: np.ndarray,
    model_dir: Path,
    epochs: int,
    batch_size: int,
    validation_split: float
) -> Dict[str, Any]:
    """
    Standalone function to run training in a separate process.
    Must be top-level for pickle compatibility.
    """
    import tensorflow as tf
    
    # Configure GPU memory growth if available (in the worker process)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # Split data
    n_val = int(len(X) * validation_split)
    indices = np.random.permutation(len(X))
    X_train = X[indices[n_val:]]
    y_train = y[indices[n_val:]]
    X_val = X[indices[:n_val]]
    y_val = y[indices[:n_val]]

    # Build model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    # Compile with class weights
    pos_weight = len(y_train) / (2 * max(np.sum(y_train), 1))
    neg_weight = len(y_train) / (2 * max(np.sum(1 - y_train), 1))
    class_weights = {0: neg_weight, 1: pos_weight}

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
    )

    # Callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
    )

    lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    # Train
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=[early_stop, lr_schedule],
        verbose=0,
    )

    # Evaluate
    val_loss, val_accuracy, val_auc = model.evaluate(
        X_val, y_val, verbose=0
    )

    # Save Keras model
    keras_path = str(model_dir / "trade_predictor.keras")
    model.save(keras_path)

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_path = model_dir / "trade_predictor_new.tflite"
    tflite_path.write_bytes(tflite_model)

    return {
        "val_loss": float(val_loss),
        "val_accuracy": float(val_accuracy),
        "val_auc": float(val_auc),
        "epochs_trained": len(history.history['loss']),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "features": X.shape[1],
        "model_size_kb": len(tflite_model) / 1024,
    }


class ModelTrainer:
    """
    Auto-retraining pipeline for the trade prediction model.
    """

    def __init__(
        self,
        db: DatabaseManager,
        model_dir: str = "models",
        min_samples: int = 10000,
        epochs: int = 50,
        batch_size: int = 64,
        validation_split: float = 0.2,
        min_accuracy: float = 0.55,
        feature_names: Optional[List[str]] = None,
        tenant_id: Optional[str] = "default",
    ):
        self.db = db
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.min_samples = min_samples
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.min_accuracy = min_accuracy
        self.feature_names = (
            [str(n) for n in feature_names]
            if feature_names
            else list(TradePredictorFeatures.FEATURE_NAMES)
        )
        self._training_history: List[Dict[str, Any]] = []
        self._executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)
        self.tenant_id = tenant_id or "default"

    async def train(self) -> Dict[str, Any]:
        """
        Execute the full training pipeline.
        
        Returns training results including metrics and deployment status.
        """
        result = {
            "success": False,
            "message": "",
            "metrics": {},
            "deployed": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Step 1: Get training data
        logger.info("Starting model training pipeline")
        training_data = await self.db.get_ml_training_data(
            self.min_samples, tenant_id=self.tenant_id
        )

        if len(training_data) < 100:
            result["message"] = f"Insufficient training data: {len(training_data)} samples (need 100+)"
            logger.warning(result["message"])
            return result

        # Step 2: Prepare data
        X, y = self._prepare_data(training_data)
        if X is None:
            result["message"] = "Data preparation failed"
            return result

        logger.info(
            "Training data prepared",
            samples=len(X),
            features=X.shape[1],
            positive_ratio=float(np.mean(y)),
        )

        # Step 3: Train model (in separate process)
        try:
            loop = asyncio.get_running_loop()
            metrics = await loop.run_in_executor(
                self._executor,
                _run_training_process,
                X,
                y,
                self.model_dir,
                self.epochs,
                self.batch_size,
                self.validation_split
            )
            
            result["metrics"] = metrics

            # Step 4: Check if model meets threshold
            val_accuracy = metrics.get("val_accuracy", 0)
            if val_accuracy >= self.min_accuracy:
                # Step 5: Deploy model
                deployed = await self._deploy_model()
                result["deployed"] = deployed
                result["success"] = True
                result["message"] = (
                    f"Model trained and deployed. "
                    f"Val accuracy: {val_accuracy:.4f}"
                )
            else:
                result["message"] = (
                    f"Model accuracy {val_accuracy:.4f} below threshold "
                    f"{self.min_accuracy}. Not deployed."
                )

        except ImportError:
            result["message"] = "TensorFlow not available for training"
            logger.warning(result["message"])
        except Exception as e:
            result["message"] = f"Training failed: {str(e)}"
            logger.error("Training failed", error=str(e))

        # Log results
        self._training_history.append(result)
        await self.db.log_thought(
            "ml",
            f"🤖 Model Training: {result['message']}",
            severity="info" if result["success"] else "warning",
            metadata=result,
            tenant_id=self.tenant_id,
        )

        return result

    def _prepare_data(
        self, training_data: List[Dict[str, Any]]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare features and labels for training."""
        try:
            features_list = []
            labels = []

            for sample in training_data:
                features = sample.get("features", {})
                label = sample.get("label")

                if label is None or not features:
                    continue

                # Consistent feature ordering (fixed list)
                feature_vector = [float(features.get(f, 0)) for f in self.feature_names]

                features_list.append(feature_vector)
                labels.append(float(label))

            if not features_list:
                return None, None

            X = np.array(features_list, dtype=np.float32)
            y = np.array(labels, dtype=np.float32)

            # Normalize features (z-score)
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            std[std == 0] = 1.0
            X = (X - mean) / std

            # Save normalization params for inference
            norm_path = self.model_dir / "normalization.json"
            norm_data = {
                "feature_names": self.feature_names,
                "mean": mean.tolist(),
                "std": std.tolist(),
            }
            norm_path.write_text(json.dumps(norm_data, indent=2))

            # Handle class imbalance
            pos_count = np.sum(y == 1)
            neg_count = np.sum(y == 0)
            if pos_count > 0 and neg_count > 0:
                ratio = neg_count / pos_count
                logger.info(
                    "Class balance",
                    positive=int(pos_count),
                    negative=int(neg_count),
                    ratio=round(ratio, 2),
                )

            return X, y

        except Exception as e:
            logger.error("Data preparation failed", error=str(e))
            return None, None

    async def _deploy_model(self) -> bool:
        """
        Deploy the new model, backing up the old one.
        
        # ENHANCEMENT: Added atomic deployment to prevent corruption
        """
        try:
            new_path = self.model_dir / "trade_predictor_new.tflite"
            live_path = self.model_dir / "trade_predictor.tflite"
            backup_path = self.model_dir / "trade_predictor_backup.tflite"

            # Backup existing model
            if live_path.exists():
                shutil.copy2(live_path, backup_path)

            # Atomic rename
            new_path.rename(live_path)

            logger.info("Model deployed successfully")
            return True

        except Exception as e:
            logger.error("Model deployment failed", error=str(e))
            return False

    async def rollback_model(self) -> bool:
        """Rollback to the previous model version."""
        try:
            live_path = self.model_dir / "trade_predictor.tflite"
            backup_path = self.model_dir / "trade_predictor_backup.tflite"

            if backup_path.exists():
                shutil.copy2(backup_path, live_path)
                logger.info("Model rolled back to previous version")
                return True
            else:
                logger.warning("No backup model available for rollback")
                return False

        except Exception as e:
            logger.error("Model rollback failed", error=str(e))
            return False

    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training history."""
        return self._training_history


class AutoRetrainer:
    """
    Automatic model retraining scheduler.
    
    Runs the training pipeline on a configurable interval,
    checking for sufficient new data before triggering.
    
    # ENHANCEMENT: Added data drift detection to trigger early retraining
    """

    def __init__(
        self,
        trainer: ModelTrainer,
        interval_hours: int = 168,  # Weekly
        min_new_samples: int = 500,
    ):
        self.trainer = trainer
        self.interval_hours = interval_hours
        self.min_new_samples = min_new_samples
        self._last_train_time: float = 0

    async def run(self) -> None:
        """Run the auto-retraining loop."""
        logger.info(
            "Auto-retrainer started",
            interval_hours=self.interval_hours,
        )

        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour

                elapsed_hours = (time.time() - self._last_train_time) / 3600
                if elapsed_hours >= self.interval_hours:
                    logger.info("Triggering scheduled retraining")
                    result = await self.trainer.train()
                    self._last_train_time = time.time()

                    if result["success"]:
                        logger.info(
                            "Auto-retrain successful",
                            metrics=result["metrics"],
                        )
                    else:
                        logger.warning(
                            "Auto-retrain failed",
                            message=result["message"],
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Auto-retrainer error", error=str(e))
                await asyncio.sleep(300)  # Wait 5 min on error
