import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models.mlp import MLP
from tensor.tensor_scratch import TensorT

original_init = TensorT.__init__

def patched_init(self, data, _op=None, _parent=()):
    # Handle TensorT objects by extracting their data
    if isinstance(data, TensorT):
        # print(f"[DEBUG] Extracting data from TensorT object")
        data = data.data
    elif isinstance(data, (float, int)):
        print(f"[DEBUG] Converting scalar {data} to 2D tensor")
        data = [[float(data)]]
    elif isinstance(data, list) and data and not isinstance(data[0], list):
        print(f"[DEBUG] Converting 1D list to 2D tensor")
        data = [data]
    
    original_init(self, data, _op=_op, _parent=_parent)

TensorT.__init__ = patched_init

# ------------------------------------------------------------------
# 1. Load dataset from local file
# ------------------------------------------------------------------
print("[DEBUG] Loading dataset...")

column_names = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude", "MedHouseVal"
]

data_path = "cal_housing/CaliforniaHousing/cal_housing.data"
df = pd.read_csv(data_path, header=None, names=column_names, sep=',')

X_np = df.iloc[:, :-1].values.T
y_np = df["MedHouseVal"].values.reshape(1, -1)

print(f"[DEBUG] Raw data shapes: X_np {X_np.shape}, y_np {y_np.shape}")

# Standardize features and target
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_np_scaled = scaler_X.fit_transform(X_np.T).T
y_np_scaled = scaler_y.fit_transform(y_np.T).T

print(f"[DEBUG] Scaled data shapes: X_np_scaled {X_np_scaled.shape}, y_np_scaled {y_np_scaled.shape}")

# Train/test split (80/20)
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X_np_scaled.T, y_np_scaled.T, test_size=0.2, random_state=42
)

X_train_np, X_test_np = X_train_np.T, X_test_np.T
y_train_np, y_test_np = y_train_np.T, y_test_np.T

print(f"Train shapes : X {X_train_np.shape}  y {y_train_np.shape}")
print(f"Test shapes  : X {X_test_np.shape}  y {y_test_np.shape}")

# ------------------------------------------------------------------
# 2. Convert to TensorT
# ------------------------------------------------------------------
print("[DEBUG] Converting to TensorT...")

X_train = TensorT(X_train_np.tolist())
print(f"[DEBUG] X_train created with shape {X_train.shape}")

y_train = TensorT(y_train_np.tolist())
print(f"[DEBUG] y_train created with shape {y_train.shape}")

X_test = TensorT(X_test_np.tolist())
print(f"[DEBUG] X_test created with shape {X_test.shape}")

y_test = TensorT(y_test_np.tolist())
print(f"[DEBUG] y_test created with shape {y_test.shape}")

# Ensure y_train and y_test have explicit 2D shape for backward compatibility
if len(y_train.shape) == 1:
    print("[DEBUG] Reshaping y_train from 1D to 2D")
    y_train = TensorT([y_train.data])
if len(y_test.shape) == 1:
    print("[DEBUG] Reshaping y_test from 1D to 2D")
    y_test = TensorT([y_test.data])

print(f"[DEBUG] Final tensor shapes: X_train {X_train.shape}, y_train {y_train.shape}")

# ------------------------------------------------------------------
# 3. Build and train regression MLP
# ------------------------------------------------------------------
print("[DEBUG] Initializing MLP...")

mlp = MLP(
    input_size=X_train.shape[0],
    hidden_layers=[16, 8],
    output_size=1,
    weight_initialization='he_normal',
    activation_func='relu',
    loss_function='mean_squared_error',
    learning_rate=0.01
)

print("[DEBUG] Starting training...")

# Monkey patch the train method to add epoch debug messages
original_train = mlp.train

def debug_train(self, X, Y, epochs, print_cost_every=100):
    print(f"\n--- Starting Training for {epochs} epochs ---")
    start_time = time.time()
    costs = []
    for i in range(epochs):
        # print(f"[DEBUG] Starting epoch {i}")
        
        # Original training step code would go here
        AL = self.forward(X)
        loss = self.cost(Y, AL)
        loss.zero_grad()
        loss.backward()
        self.update_parameters()
        
        # print(f"[DEBUG] Completed epoch {i}")
        
        if i % print_cost_every == 0 or i == epochs - 1:
            cost = loss.data[0][0] if hasattr(loss, 'data') else float(loss)
            costs.append(cost)
            print(f"Epoch {i:>4} | Cost: {cost:.6f}")
    
    dt = time.time() - start_time
    print(f"--- Training Finished in {dt:.2f} seconds ---")
    return costs

# Import time module
import time

try:
    costs = debug_train(mlp, X_train, y_train, epochs=250, print_cost_every=100)
    print("[DEBUG] Training completed successfully")
except Exception as e:
    print(f"[DEBUG] Training failed with error: {e}")
    import traceback
    traceback.print_exc()
    exit()

# ------------------------------------------------------------------
# 4. Comprehensive Evaluation
# ------------------------------------------------------------------
pred_scaled = mlp.predict(X_test)
pred_np_scaled = np.array(pred_scaled.data)

# Inverse transform to original scale
pred_np = scaler_y.inverse_transform(pred_np_scaled.T).T
y_test_np = scaler_y.inverse_transform(np.array(y_test.data).T).T

# Calculate multiple regression metrics
def regression_metrics(y_true, y_pred):
    # Flatten arrays
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Mean Absolute Error
    mae = np.mean(np.abs(y_true_flat - y_pred_flat))
    
    # Mean Squared Error
    mse = np.mean((y_true_flat - y_pred_flat) ** 2)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # R-squared (coefficient of determination)
    ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
    ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true_flat - y_pred_flat) / y_true_flat)) * 100
    
    return mae, mse, rmse, r2, mape

mae, mse, rmse, r2, mape = regression_metrics(y_test_np, pred_np)

print(f"\n=== Model Performance Metrics ===")
print(f"Mean Absolute Error (MAE):     ${mae:,.2f}")
print(f"Mean Squared Error (MSE):      {mse:,.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
print(f"R-squared (R²):                {r2:.4f}")
print(f"Mean Absolute Percentage Error: {mape:.2f}%")

# # ------------------------------------------------------------------
# # 5. Test Individual Predictions
# # ------------------------------------------------------------------
# def test_sample_predictions(X_test, y_test, predictions, scaler_y, num_samples=10):
#     print(f"\n=== Sample Predictions Comparison ===")
    
#     # Get original unscaled test data for interpretation
#     X_test_orig = scaler_X.inverse_transform(X_test.data)
#     y_test_orig = y_test.flatten()
#     pred_orig = predictions.flatten()
    
#     # Select random samples to display
#     indices = np.random.choice(len(y_test_orig), num_samples, replace=False)
    
#     print(f"{'Index':<6} {'True Price':<12} {'Predicted':<12} {'Error':<12} {'% Error':<10}")
#     print("-" * 62)
    
#     for i in indices:
#         true_price = y_test_orig[i]
#         pred_price = pred_orig[i] 
#         error = abs(true_price - pred_price)
#         pct_error = (error / true_price) * 100
        
#         print(f"{i:<6} ${true_price:<11,.0f} ${pred_price:<11,.0f} ${error:<11,.0f} {pct_error:<9.1f}%")

# test_sample_predictions(X_test, y_test_np, pred_np, scaler_y)


# # ------------------------------------------------------------------
# # 6. Residual Analysis
# # ------------------------------------------------------------------
# def plot_residuals(y_true, y_pred):
#     residuals = y_true.flatten() - y_pred.flatten()
    
#     plt.figure(figsize=(12, 4))
    
#     # Residual vs Predicted
#     plt.subplot(1, 2, 1)
#     plt.scatter(y_pred.flatten(), residuals, alpha=0.6)
#     plt.axhline(y=0, color='r', linestyle='--')
#     plt.xlabel('Predicted Values')
#     plt.ylabel('Residuals')
#     plt.title('Residuals vs Predicted')
#     plt.grid(True)
    
#     # Histogram of Residuals
#     plt.subplot(1, 2, 2)
#     plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
#     plt.xlabel('Residuals')
#     plt.ylabel('Frequency')
#     plt.title('Distribution of Residuals')
#     plt.grid(True)
    
#     plt.tight_layout()
#     plt.show()
    
#     # Statistical tests on residuals
#     print(f"\n=== Residual Analysis ===")
#     print(f"Residual Mean: ${np.mean(residuals):,.2f} (should be ~0)")
#     print(f"Residual Std:  ${np.std(residuals):,.2f}")
#     print(f"Min Residual:  ${np.min(residuals):,.2f}")
#     print(f"Max Residual:  ${np.max(residuals):,.2f}")

# plot_residuals(y_test_np, pred_np)

