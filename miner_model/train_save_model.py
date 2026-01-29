import numpy as np
import pandas as pd
import tensorflow as tf

# ---- Create synthetic time series data ----
timesteps = 200
values = np.sin(np.linspace(0, 20, timesteps)) * 10 + 100

df = pd.DataFrame({
    "timestamp": np.arange(timesteps),
    "value": values
})

df.to_csv("my_data.csv", index=False)

# ---- Prepare data for LSTM ----
SEQ_LEN = 10
X, y = [], []

for i in range(len(values) - SEQ_LEN):
    X.append(values[i:i + SEQ_LEN])
    y.append(values[i + SEQ_LEN])

X = np.array(X).reshape(-1, SEQ_LEN, 1)
y = np.array(y)

# ---- Define simple LSTM model ----
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(16, input_shape=(SEQ_LEN, 1)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.fit(X, y, epochs=1, batch_size=16)

# ---- Save model ----
model.save("my_model.h5")

print("Saved my_model.h5 and my_data.csv")
