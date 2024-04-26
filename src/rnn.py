import torch
import torch.nn as nn
import numpy as np

# Generate some example data
def generate_data(n_points):
    X = np.random.randn(n_points, 1)  # Input features (random for this example)
    y = np.sin(X)  # Target values (sine function of input)
    return X, y

# Define the RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

# Hyperparameters
input_size = 1
hidden_size = 50
output_size = 1
learning_rate = 0.01
num_epochs = 10
batch_size = 32

# Generate training data
X_train, y_train = generate_data(1000)
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(2)  # Add a dimension for sequence length
y_train = torch.tensor(y_train, dtype=torch.float32)


# Initialize model, loss function, and optimizer
model = RNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for i in range(0, len(X_train), batch_size):
        inputs = X_train[i:i+batch_size]
        targets = y_train[i:i+batch_size]

        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')



if __name__ == "__main__":
    X_test, y_test = generate_data(10)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(2)

    with torch.no_grad():
        predictions = model(X_test)

    for i in range(len(predictions)):
        print("Predicted:", predictions[i][0], "| Actual:", y_test[i][0])


