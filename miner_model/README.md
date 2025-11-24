# Miner Model Plugin - Simple Guide

## 🚀 Quick Start (3 Steps!)

### Step 1: Train Your Model in Notebook
Train your model as usual in your notebook. When done, save it:
```python
model.save('my_model.h5')
```

### Step 2: Create Your Model File
1. Copy `student_models/template.py` to `student_models/my_model.py`
2. Fill in **only 2 sections**:
   - **SECTION 1**: Load your model (copy from notebook)
   - **SECTION 2**: Load your data (copy from notebook)  


### Step 3: Run the Miner
```bash
python -m miner_model.miner_plugin \
  --netuid 420 \
  --subtensor.network test \
  --wallet.name YOUR_MINER_NAME \
  --wallet.hotkey YOUR_HOTKEY_NAME
```

That's it! The miner will automatically find and use your model.

---

## 📝 Example: LSTM Model

See `student_models/lstm_example.py` for a complete example.

**Your model file should look like this:**

```python
# student_models/my_model.py
from tensorflow.keras.models import load_model
import pandas as pd
from .helpers import predict_1hour_ahead, prepare_dataframe

# SECTION 1: Load Your Model
model = load_model('my_model.h5')

# SECTION 2: Load Your Data
df = pd.read_csv('my_data.csv')
data = prepare_dataframe(df)  # Helper handles formatting!

# SECTION 3: Predict Function (ALREADY DONE!)
def predict(timestamp):
    return predict_1hour_ahead(model, data, timestamp)
```

That's it! The helper function handles the standard 1-hour-ahead prediction pattern.

---

## 📋 Helper Functions

The template uses helper functions that handle the standard prediction pattern:

- **`predict_1hour_ahead()`** - Complete 1-hour-ahead prediction function
- **`prepare_dataframe()`** - Automatically formats your CSV data
- **`get_recent_prices()`** - Gets historical prices for prediction
- **`calculate_interval()`** - Calculates confidence intervals

You can customize the predict function if needed:
```python
def predict(timestamp):
    return predict_1hour_ahead(
        model, data, timestamp,
        n_steps=12,  # Change if different time interval
        interval_method='std',  # or 'fixed'
        interval_std=0.002586  # Your standard error
    )
```

---

## 🗂️ File Structure

```
miner_model/
├── student_models/          ← Put your model files here!
│   ├── template.py         ← Copy this to create your model
│   └── lstm_example.py     ← Example LSTM model
├── miner_plugin.py         ← Run this to start miner
└── README.md               ← This file
```

---

## ❓ Common Questions

**Q: Where do I put my model file?**  
A: In the `student_models/` folder. Copy `template.py` and rename it.

**Q: How do I load my model?**  
A: Copy your model loading code from the notebook into SECTION 1.

**Q: How do I handle data paths?**  
A: Use relative paths from your model file, or absolute paths.

**Q: What if my prediction fails?**  
A: Return `(None, None)` - the miner will handle it gracefully.

**Q: Can I use multiple models?**  
A: The miner uses the first model file found in `student_models/`. Put only one model file there.

---

## 🐛 Troubleshooting

**"No student model found"**  
→ Make sure you created a file in `student_models/` folder (not just `template.py`)

**"Model file does not have a predict() function"**  
→ The template already includes the predict function. Make sure you didn't delete it!

**"Failed to load model"**  
→ Check that your model file path is correct in SECTION 1

**"Prediction returned None"**  
→ Check your data path in SECTION 2, and make sure you have enough historical data

---

## 📚 Need More Help?

- See `student_models/lstm_example.py` for a complete working example
- Check `student_models/template.py` for the template structure
- Look at your notebook - most code can be copied directly!

---

## 🎉 That's It!

You're ready to run your miner. Just:
1. Create your model file in `student_models/`
2. Fill in **only 2 sections** (model and data loading)
3. Run `python -m miner_model.miner_plugin`

The predict function is already done - it uses standard helpers! 🚀
