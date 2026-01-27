# Miner Model Plugin - Simple Guide

## 🚀 Quick Start

### Step 1: Fork the Repository
1. Go to the original repository on GitHub
2. Click the "Fork" button to create your own copy
3. This gives you your own version to work with

### Step 2: Clone to Your Personal Device
```bash
git clone https://github.com/YOUR_USERNAME/bittbridge.git
cd bittbridge/bittbridge/miner_model
```

### Step 3: Place Your Model Files
Train your model in your notebook and save it:
```python
model.save('my_model.h5')
```

Then place both files anywhere in the `miner_model/` directory (or create your own subdirectory like `my_models/`):
- Your `.h5` model file
- Your `.csv` data file

The `student_models/my_model.py` file is already set up and ready to use! It will automatically find both files - no configuration needed!

**Note:** The `LSTM_outside_example/` directory is just an example showing how your custom model workflow could look like. Your actual model and data files should be placed elsewhere in the `miner_model/` directory (the `my_model.py` will automatically find `.h5` and `.csv` files, excluding `LSTM_outside_example`).

### Step 4: Push Your Changes Back
```bash
git add .
git commit -m "Add my model"
git push origin main
```
### Step 5: OPTIONAL Deploy it on Google Cloud Platform Virtual Machine

- Use your personal GCP account (free credits) so it runs 24/7
- Spin up a VM (Ubuntu-based)
- Deploy:
    - miner
    - validator
- Open required ports

### Step 6: Run the Miner & Validator
```bash
python -m miner_model.miner_plugin \
  --netuid 420 \
  --subtensor.network test \
  --wallet.name YOUR_MINER_NAME \
  --wallet.hotkey YOUR_HOTKEY_NAME
```
```bash
# Validator
# In the terminal where you will start validator paste these commands:

# Set the variable `COINGECKO_API_KEY` in your environment:
export COINGECKO_API_KEY="PASTE_YOUR_COINGECKO_API_KEY_HERE"

# Set the variable `WANDB_API_KEY` in your environment:
export WANDB_API_KEY="PASTE_YOUR_API_KEY"

# Run validator 
python3 -m neurons.validator \
  --netuid 420 \
  --subtensor.network test \
  --wallet.name YOUR_VALIDATOR_NAME \
  --wallet.hotkey YOUR_VALIDATOR_HOTKEY_NAME \
  --logging.debug
```

That's it! The miner will automatically find and use your model.

**Remember:** After setting up your model, commit and push your changes to your forked repository!


---

## 📝 How It Works

The `student_models/my_model.py` file automatically discovers and loads your files:

**Note:** `LSTM_outside_example/` is just an example showing how your custom model workflow could look like. It demonstrates organizing model files, data files, and notebooks in a separate directory. Your actual model and data files should be placed elsewhere in `miner_model/` (the `my_model.py` will find them automatically).

**The `my_model.py` file:**
- Automatically searches for `.h5` files in `miner_model/` directory (excluding `LSTM_outside_example`)
- Automatically searches for `.csv` files in `miner_model/` directory (excluding `LSTM_outside_example`)
- Uses the first file found (you can customize this if you have multiple files)
- Includes a ready-to-use `predict()` function

**No configuration needed!** Just place your `.h5` and `.csv` files in `miner_model/` and run the miner. The helper function handles the standard 1-hour-ahead prediction pattern.
---
## 🗂️ File Structure

```
miner_model/
├── student_models/          ← Model Python files
│   ├── my_model.py         ← Default model file (ready to use!)
│   └── helpers.py          ← Helper functions (don't modify)
├── LSTM_outside_example/   ← Example workflow (just for reference)
│   ├── lstm_model.h5       ← Example model file
│   └── USDT-CNY_scraper (2).csv  ← Example data
├── my_model.h5             ← Your model file (can be anywhere in miner_model/)
├── my_data.csv             ← Your data file (or in a subdirectory)
├── miner_plugin.py         ← Run this to start miner
└── README.md               ← This file
```

**Note:** You can organize your files however you like! The `my_model.py` automatically searches for both `.h5` and `.csv` files in `miner_model/` and all its subdirectories (except `LSTM_outside_example`). For example, you could create:
- `miner_model/my_models/my_model.h5`
- `miner_model/data/my_data.csv`
- `miner_model/my_model.h5` (root level)
- `miner_model/my_data.csv` (root level)

All will work! The `my_model.py` will find them automatically.

---

## ❓ Common Questions

**Q: Where do I put my model and data files?**  
A: Put your `.h5` model file and `.csv` data file anywhere in the `miner_model/` directory (or any subdirectory). The `my_model.py` file will automatically find both! The `my_model.py` file is already in `student_models/` and ready to use.

**Q: How do I load my model and data?**  
A: The `my_model.py` automatically searches for both `.h5` and `.csv` files in `miner_model/` directory. Just make sure your files are there - no configuration needed! Both model and data loading are handled automatically.

**Q: What about the LSTM_outside_example directory?**  
A: That's just an example showing how you could organize your model workflow. Your actual model and data files should be placed elsewhere in `miner_model/` (the `my_model.py` excludes `LSTM_outside_example` when searching for files).

**Q: Can I customize the file paths?**  
A: Yes! If you have multiple `.h5` or `.csv` files and want to select a specific one, you can customize `model_path` or `data_path` in SECTION 1 or SECTION 2 of `my_model.py`.

**Q: What if my prediction fails?**  
A: Return `(None, None)` - the miner will handle it gracefully.

**Q: Can I use multiple models?**  
A: The miner uses the first Python model file found in `student_models/`. The default `my_model.py` is ready to use. If you have multiple `.h5` files, `my_model.py` uses the first one found (you can customize this in SECTION 1 if needed).

---

## 🐛 Troubleshooting

**"No student model found"**  
→ Make sure `my_model.py` exists in `student_models/` folder. If you deleted it, restore it from the repository.

**"Model file does not have a predict() function"**  
→ The `my_model.py` already includes the predict function. Make sure you didn't delete it!

**"Failed to load model"**  
→ Make sure you have a `.h5` file in the `miner_model/` directory. The `my_model.py` searches automatically, but you can customize `model_path` in SECTION 1 if needed.

**"No .csv data files found"**  
→ Make sure you have a `.csv` file in the `miner_model/` directory. The `my_model.py` searches automatically, but you can customize `data_path` in SECTION 2 if needed.

**"Prediction returned None"**  
→ Check that your data file has enough historical data and is properly formatted

---

## 📚 Need More Help?

- Check `student_models/my_model.py` to see how it works
- Look at `LSTM_outside_example/` to see an example workflow structure
- Look at your notebook - most code can be copied directly!

---

