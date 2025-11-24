# Quick Start: LSTM Model Integration

## 🚀 5-Minute Quick Start

### Step 1: Save Your Trained Model

Add this to your notebook after training:

```python
# In your notebook, after training the LSTM model
model.save('lstm_model.h5')
```

Make sure the file is saved in: `miner_model/outside_model/lstm_model.h5`

### Step 2: Test Your Model

```bash
cd /Users/dmitrii/Desktop/miner_plugin/bittbridge
python miner_model/outside_model/test_lstm_model.py
```

### Step 3: Run Your Miner

Create `run_lstm_miner.py` in the project root:

```python
from miner_model.miner_plugin import Miner
from miner_model.example_models.lstm_model import LSTMModel

if __name__ == "__main__":
    model = LSTMModel()
    with Miner(model=model) as miner:
        import time
        import bittensor as bt
        bt.logging.info("LSTM Miner started...")
        while True:
            time.sleep(5)
```

Then run:

```bash
python run_lstm_miner.py \
  --netuid 420 \
  --subtensor.network test \
  --wallet.name YOUR_MINER_NAME \
  --wallet.hotkey YOUR_MINER_HOTKEY_NAME
```

---

## 📁 File Structure

```
miner_model/
├── example_models/
│   ├── lstm_model.py          ← Your LSTM model class (already created!)
│   └── simple_model.py
├── outside_model/
│   ├── lstm_model.h5          ← Your trained model (you need to save this)
│   ├── USDT-CNY_scraper (2).csv  ← Your data (already exists)
│   ├── test_lstm_model.py     ← Test script (already created!)
│   ├── WORKFLOW_GUIDE.md      ← Detailed guide
│   └── QUICK_START.md        ← This file
└── miner_plugin.py
```

---

## ✅ Checklist

- [ ] Save trained model: `model.save('lstm_model.h5')`
- [ ] Test model: `python miner_model/outside_model/test_lstm_model.py`
- [ ] Create runner script (or modify `miner_plugin.py`)
- [ ] Run miner on testnet
- [ ] Monitor logs for predictions

---

## 🔧 Common Issues

**"Model file not found"**
→ Save your model: `model.save('lstm_model.h5')` in the notebook

**"Insufficient historical data"**
→ Make sure your CSV has data before the prediction timestamp

**"TensorFlow not available"**
→ Install: `pip install tensorflow`

---

## 📖 Full Documentation

See `WORKFLOW_GUIDE.md` for detailed instructions and troubleshooting.

