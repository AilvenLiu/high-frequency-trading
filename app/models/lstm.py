import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from datetime import datetime
import json
import os

@dataclass
class PredictorConfig:
    """Model configuration"""
    input_size: int = 8  # Number of features
    hidden_size: int = 64
    num_layers: int = 2
    output_size: int = 3  # Trend directions: up, down, sideways
    sequence_length: int = 20
    batch_size: int = 32
    learning_rate: float = 0.001
    dropout: float = 0.2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
class AttentionLayer(nn.Module):
    """Attention mechanism for time series"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
        Returns:
            context: (batch_size, hidden_size)
            attention_weights: (batch_size, seq_len)
        """
        attention_weights = self.attention(hidden_states)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(
            attention_weights.squeeze(-1),
            dim=1
        )  # (batch, seq_len)
        
        context = torch.bmm(
            attention_weights.unsqueeze(1),
            hidden_states
        ).squeeze(1)  # (batch, hidden_size)
        
        return context, attention_weights
        
class LSTMPredictor(nn.Module):
    """LSTM with Attention for time series prediction"""
    
    def __init__(self, config: PredictorConfig):
        super().__init__()
        self.config = config
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = AttentionLayer(config.hidden_size)
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.output_size)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                
    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            output: (batch_size, output_size)
            attention_weights: (batch_size, seq_len)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        
        # Apply attention
        context, attention_weights = self.attention(lstm_out)
        
        # Final prediction
        output = self.fc(context)
        
        return output, attention_weights
        
class ModelTrainer:
    """Model trainer and predictor"""
    
    def __init__(
        self,
        config: PredictorConfig,
        model_dir: str = "models/saved"
    ):
        self.config = config
        self.model_dir = model_dir
        self.device = torch.device(config.device)
        
        # Initialize model
        self.model = LSTMPredictor(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate
        )
        
        # Initialize state
        self.feature_scaler = None
        self._create_model_dir()
        
    def _create_model_dir(self):
        """Create model directory if not exists"""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
    def train(
        self,
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        valid_data: Optional[torch.Tensor] = None,
        valid_labels: Optional[torch.Tensor] = None,
        num_epochs: int = 100,
        patience: int = 10
    ) -> Dict:
        """Train the model"""
        try:
            best_loss = float('inf')
            patience_counter = 0
            training_history = []
            
            for epoch in range(num_epochs):
                self.model.train()
                total_loss = 0
                num_batches = 0
                
                # Create batches
                for i in range(0, len(train_data), self.config.batch_size):
                    batch_data = train_data[i:i + self.config.batch_size]
                    batch_labels = train_labels[i:i + self.config.batch_size]
                    
                    # Move to device
                    batch_data = batch_data.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    outputs, _ = self.model(batch_data)
                    loss = self.criterion(outputs, batch_labels)
                    
                    # Backward pass
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                avg_loss = total_loss / num_batches
                
                # Validation
                if valid_data is not None and valid_labels is not None:
                    valid_loss = self._validate(valid_data, valid_labels)
                    
                    # Early stopping
                    if valid_loss < best_loss:
                        best_loss = valid_loss
                        patience_counter = 0
                        self._save_model()
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= patience:
                        logging.info(f"Early stopping at epoch {epoch}")
                        break
                        
                    training_history.append({
                        'epoch': epoch,
                        'train_loss': avg_loss,
                        'valid_loss': valid_loss
                    })
                    
                else:
                    training_history.append({
                        'epoch': epoch,
                        'train_loss': avg_loss
                    })
                    
            return training_history
            
        except Exception as e:
            logging.error(f"Error during training: {e}")
            raise
            
    def predict(
        self,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make predictions"""
        try:
            self.model.eval()
            with torch.no_grad():
                features = features.to(self.device)
                predictions, attention_weights = self.model(features)
                probabilities = torch.softmax(predictions, dim=1)
                return probabilities, attention_weights
                
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise
            
    def _validate(
        self,
        valid_data: torch.Tensor,
        valid_labels: torch.Tensor
    ) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(valid_data), self.config.batch_size):
                batch_data = valid_data[i:i + self.config.batch_size]
                batch_labels = valid_labels[i:i + self.config.batch_size]
                
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs, _ = self.model(batch_data)
                loss = self.criterion(outputs, batch_labels)
                
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / num_batches
        
    def _save_model(self):
        """Save model state"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(
            self.model_dir,
            f"model_state_{timestamp}.pth"
        )
        config_path = os.path.join(
            self.model_dir,
            f"config_{timestamp}.json"
        )
        
        # Save model state
        torch.save(self.model.state_dict(), model_path)
        
        # Save configuration
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f)
            
    def load_model(self, model_path: str, config_path: str):
        """Load model state"""
        try:
            # Load configuration
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
                self.config = PredictorConfig(**config_dict)
                
            # Initialize and load model
            self.model = LSTMPredictor(self.config).to(self.device)
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
            
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise 