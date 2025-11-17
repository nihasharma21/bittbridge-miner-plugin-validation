# Miner Model Plugin Guide

This directory contains a working example that shows how to plug predictive model into the Bittbridge subnet and run it.

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Understanding the Architecture](#understanding-the-architecture)
- [Step-by-Step Integration Guide](#step-by-step-integration-guide)
- [Model Interface Requirements](#model-interface-requirements)
- [Example: Creating Your Own Model](#example-creating-your-own-model)
- [Testing Your Model](#testing-your-model)
- [Running the Miner](#running-the-miner)
- [Common Issues and Solutions](#common-issues-and-solutions)
- [Best Practices](#best-practices)

---

## Overview

The Bittbridge subnet rewards miners for accurate USDT/CNY price predictions. This plugin system:

1. **Separate model logic** from network infrastructure
2. **Show how to implement prediction algorithm** (ML models, statistical models, API-based, etc.)
3. **Focus on your model prediction** while handling network communication

### Key Components

- **`model_interface.py`**: Abstract base class defining the model contract
- **`miner_plugin.py`**: Main miner file that integrates your model with the network
- **`example_models/simple_model.py`**: Working example implementation
- **`README.md`**: This guide

---

## Quick Start

### 1. Install Dependencies

```bash
# From the project root
pip install -e .

# Install additional model dependencies
pip install -r miner_model/requirements.txt
```

### 2. Set Up Environment Variables

```bash
export COINGECKO_API_KEY="your_api_key_here"
```

### 3. Run the Example Miner

```bash
# From the bittbridge directory
python -m miner_model.miner_plugin \
  --netuid 420 \
  --subtensor.network test \
  --wallet.name YOUR_MINER_NAME \
  --wallet.hotkey YOUR_MINER_HOTKEY_NAME \
  --logging.debug
```

The miner will use `SimpleAPIModel` by default, which fetches current prices from CoinGecko.

---

## Understanding the Architecture

### How It Works

```
┌─────────────────┐
│   Validator     │
│  (Bittensor)    │
└────────┬────────┘
         │ Challenge synapse (timestamp)
         ▼
┌─────────────────┐
│  Miner Plugin   │  ← Handles network communication
│  (miner_plugin) │
└────────┬────────┘
         │ Calls predict(timestamp)
         ▼
┌─────────────────┐
│  Your Model     │  ← Your prediction logic
│  (Your Code)    │
└────────┬────────┘
         │ Returns (prediction, interval)
         ▼
┌─────────────────┐
│  Miner Plugin   │  ← Attaches to synapse
└────────┬────────┘
         │ Response synapse
         ▼
┌─────────────────┐
│   Validator     │  ← Scores your prediction
└─────────────────┘
```

### Separation of Concerns

- **Miner Plugin**: Handles Bittensor network communication, request filtering, logging
- **Your Model**: Focuses solely on generating accurate predictions
- **Interface**: Clean contract between miner and model

---

## Step-by-Step Integration Guide

### Step 1: Understand the Model Interface

Your model must implement the `PredictionModel` interface defined in `model_interface.py`. The key method is:

```python
def predict(self, timestamp: str) -> Tuple[Optional[float], Optional[List[float]]]:
    """
    Generate a USDT/CNY price prediction.
    
    Returns:
        (prediction, interval):
        - prediction: float or None (predicted price)
        - interval: [lower, upper] or None (90% confidence interval)
    """
    pass
```

### Step 2: Create Your Model Class

Create a new file (e.g., `my_model.py`) and implement your model:

```python
from miner_model.model_interface import PredictionModel
from typing import Tuple, Optional, List

class MyCustomModel(PredictionModel):
    def __init__(self):
        # Initialize your model here
        # Load weights, connect to APIs, etc.
        pass
    
    def predict(self, timestamp: str) -> Tuple[Optional[float], Optional[List[float]]]:
        # Your prediction logic here
        prediction = 7.25  # Your predicted price
        interval = [7.10, 7.40]  # 90% confidence interval
        return prediction, interval
```

### Step 3: Integrate with Miner

Modify `miner_plugin.py` to use your model:

```python
# At the bottom of miner_plugin.py, replace:
from my_model import MyCustomModel

model = MyCustomModel()
miner = Miner(model=model)
```

Or pass it when running:

```python
# In your own script
from miner_model.miner_plugin import Miner
from my_model import MyCustomModel

model = MyCustomModel()
miner = Miner(model=model)
miner.run()
```

### Step 4: Test Your Model

See [Testing Your Model](#testing-your-model) section below.

### Step 5: Run the Miner

See [Running the Miner](#running-the-miner) section below.

---

## Model Interface Requirements

### Required Methods

#### `predict(timestamp: str) -> Tuple[Optional[float], Optional[List[float]]]`

**Purpose**: Generate a USDT/CNY price prediction for the given timestamp.

**Input**:
- `timestamp`: ISO format string (e.g., `"2024-01-15T10:30:00+00:00"`)

**Output**:
- `prediction`: `float` or `None` - The predicted USDT/CNY price
- `interval`: `[lower, upper]` or `None` - 90% confidence interval

**Important Notes**:
- Return `(None, None)` if prediction fails (API error, insufficient data, etc.)
- Validators ignore `None` predictions (miner gets zero reward)
- Interval should represent 90% confidence: `[lower_bound, upper_bound]`
- Predictions should be reasonable (USDT/CNY typically 6-8 range)

### Optional Methods

#### `initialize() -> bool`

Called when miner starts. Use for:
- Loading pre-trained weights
- Connecting to external services
- Validating configuration

Return `True` if successful, `False` otherwise.

#### `cleanup() -> None`

Called when miner shuts down. Use for:
- Saving state
- Closing connections
- Releasing resources

---

## Example: Creating Your Own Model

### Example 1: Simple API-Based Model

```python
import requests
from miner_model.model_interface import PredictionModel
from typing import Tuple, Optional, List

class MyAPIModel(PredictionModel):
    def __init__(self):
        self.api_key = os.getenv("MY_API_KEY")
    
    def predict(self, timestamp: str) -> Tuple[Optional[float], Optional[List[float]]]:
        try:
            # Fetch current price from your API
            response = requests.get("https://api.example.com/price")
            price = response.json()["usdt_cny"]
            
            # Estimate interval (naive approach)
            interval = [price * 0.99, price * 1.01]
            
            return price, interval
        except Exception as e:
            return None, None
```

### Example 2: Machine Learning Model

```python
import numpy as np
import pandas as pd
from miner_model.model_interface import PredictionModel
from typing import Tuple, Optional, List

class MyMLModel(PredictionModel):
    def __init__(self):
        # Load your trained model
        self.model = self._load_model("path/to/model.pkl")
        self.scaler = self._load_scaler("path/to/scaler.pkl")
    
    def initialize(self) -> bool:
        # Validate model is loaded correctly
        return self.model is not None
    
    def predict(self, timestamp: str) -> Tuple[Optional[float], Optional[List[float]]]:
        try:
            # Prepare features from timestamp and historical data
            features = self._prepare_features(timestamp)
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            
            # Estimate uncertainty (from your model or historical errors)
            std_error = 0.02  # Example: 2% standard error
            interval = [
                prediction - 1.64 * std_error * prediction,
                prediction + 1.64 * std_error * prediction
            ]
            
            return float(prediction), interval
        except Exception as e:
            return None, None
    
    def _load_model(self, path):
        # Your model loading logic
        pass
    
    def _prepare_features(self, timestamp):
        # Your feature engineering logic
        pass
```

### Example 3: Time Series Forecasting Model

```python
from miner_model.model_interface import PredictionModel
from typing import Tuple, Optional, List
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

class TimeSeriesModel(PredictionModel):
    def __init__(self):
        self.historical_data = []
        self.model = None
    
    def predict(self, timestamp: str) -> Tuple[Optional[float], Optional[List[float]]]:
        try:
            # Update model with latest data
            if len(self.historical_data) > 100:
                self.model = ARIMA(self.historical_data, order=(1, 1, 1))
                self.model = self.model.fit()
            
            # Forecast 1 hour ahead
            forecast = self.model.forecast(steps=1)
            prediction = float(forecast[0])
            
            # Get confidence interval from model
            conf_int = self.model.get_forecast(steps=1).conf_int()
            interval = [float(conf_int.iloc[0, 0]), float(conf_int.iloc[0, 1])]
            
            return prediction, interval
        except Exception as e:
            return None, None
```

---

## Testing Your Model

### Unit Testing

Test your model independently before integrating:

```python
from my_model import MyCustomModel

def test_my_model():
    model = MyCustomModel()
    
    # Test initialization
    assert model.initialize() == True
    
    # Test prediction
    timestamp = "2024-01-15T10:30:00+00:00"
    prediction, interval = model.predict(timestamp)
    
    # Validate outputs
    assert prediction is not None, "Prediction should not be None"
    assert isinstance(prediction, float), "Prediction should be float"
    assert 6.0 <= prediction <= 8.0, "Prediction should be reasonable"
    
    if interval is not None:
        assert len(interval) == 2, "Interval should be [lower, upper]"
        assert interval[0] < interval[1], "Lower bound should be less than upper"
        assert interval[0] <= prediction <= interval[1], "Prediction should be in interval"
    
    print("✅ All tests passed!")

if __name__ == "__main__":
    test_my_model()
```

### Integration Testing

Test with the miner in a local environment:

```python
from miner_model.miner_plugin import Miner
from my_model import MyCustomModel
import bittbridge

# Create a mock challenge
challenge = bittbridge.protocol.Challenge(
    timestamp="2024-01-15T10:30:00+00:00"
)

# Test miner with your model
model = MyCustomModel()
miner = Miner(model=model)

# Test forward method
import asyncio
result = asyncio.run(miner.forward(challenge))

assert result.prediction is not None
assert result.interval is not None
print("✅ Integration test passed!")
```

---

## Running the Miner

### Basic Usage

```bash
python -m miner_model.miner_plugin \
  --netuid 420 \
  --subtensor.network test \
  --wallet.name YOUR_MINER_NAME \
  --wallet.hotkey YOUR_MINER_HOTKEY_NAME
```

### With Custom Model

If you've modified `miner_plugin.py` to use your model, it will automatically use it. Otherwise, create a custom script:

```python
# run_my_miner.py
from miner_model.miner_plugin import Miner
from my_model import MyCustomModel

if __name__ == "__main__":
    model = MyCustomModel()
    miner = Miner(model=model)
    miner.run()
```

Then run:
```bash
python run_my_miner.py \
  --netuid 420 \
  --subtensor.network test \
  --wallet.name YOUR_MINER_NAME \
  --wallet.hotkey YOUR_MINER_HOTKEY_NAME
```

### Command-Line Arguments

Common arguments:
- `--netuid 420`: Subnet ID
- `--subtensor.network test`: Network (test/mainnet)
- `--wallet.name NAME`: Wallet name
- `--wallet.hotkey HOTKEY`: Hotkey name
- `--logging.debug`: Enable debug logging
- `--axon.port 8091`: Port for incoming connections

See the main README.md for full configuration options.

---

## Common Issues and Solutions

### Issue: Model Returns None Predictions

**Symptoms**: Logs show "Model returned None prediction"

**Solutions**:
- Check API keys are set: `echo $COINGECKO_API_KEY`
- Verify API connectivity: Test your API calls independently
- Add error handling: Catch exceptions and log them
- Check data availability: Ensure you have sufficient historical data

### Issue: Import Errors

**Symptoms**: `ModuleNotFoundError` when importing your model

**Solutions**:
- Ensure your model file is in Python path
- Use absolute imports: `from miner_model.model_interface import PredictionModel`
- Install dependencies: `pip install -r requirements.txt`

### Issue: Predictions Are Always Wrong

**Symptoms**: Low rewards from validators

**Solutions**:
- Verify you're predicting future price (1 hour ahead), not current price
- Check your model is using the timestamp correctly
- Validate your data sources are accurate
- Test your model independently before deploying

### Issue: Miner Not Receiving Requests

**Symptoms**: No logs showing incoming challenges

**Solutions**:
- Check miner is registered: `btcli wallet overview --wallet.name YOUR_MINER`
- Verify network connectivity (see main README troubleshooting)
- Check firewall/port forwarding
- Ensure validator is running and querying miners

### Issue: Interval Format Errors

**Symptoms**: Validator rejects responses

**Solutions**:
- Ensure interval is `[lower, upper]` list format
- Verify `lower < upper`
- Check interval represents 90% confidence
- Return `None` if you can't estimate interval

---

## Best Practices

### 1. Error Handling

Always handle errors gracefully:

```python
def predict(self, timestamp: str) -> Tuple[Optional[float], Optional[List[float]]]:
    try:
        # Your prediction logic
        return prediction, interval
    except Exception as e:
        bt.logging.error(f"Prediction failed: {e}")
        return None, None  # Don't crash the miner
```

### 2. Logging

Use appropriate log levels:

```python
bt.logging.debug("Detailed debugging info")
bt.logging.info("General information")
bt.logging.warning("Something unexpected but handled")
bt.logging.error("Errors that need attention")
```

### 3. Performance

- Cache API responses when possible
- Pre-load models in `initialize()`
- Use async operations for I/O if needed
- Keep prediction latency low (< 1 second)

### 4. Testing

- Test your model independently
- Validate predictions are reasonable
- Test error cases (API failures, missing data)
- Run integration tests with miner

### 5. Documentation

- Document your model's approach
- Explain any assumptions
- Note required environment variables
- Include example usage

### 6. Model Updates

- Version your models
- Log model version in predictions
- Test new models before deploying
- Keep old models for rollback

---

## Next Steps

1. ✅ Review the example model (`example_models/simple_model.py`)
2. ✅ Understand the interface (`model_interface.py`)
3. ✅ Create your own model
4. ✅ Test your model
5. ✅ Integrate with miner
6. ✅ Deploy and monitor

## Getting Help

- Check the main project README.md
- Review example models in `example_models/`
- Examine the model interface documentation
- Test with the simple model first

---

## License

This code is licensed under the MIT License. See the main project LICENSE file for details.

