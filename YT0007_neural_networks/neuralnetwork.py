import numpy as np


def generate_data(n: int):
    inputs = np.random.randint(0, 11, size=(n, 3))
    output = np.array([2 * a - 3 * b + 0.5 * c for a, b, c in inputs])
    return inputs, output

# Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# ReLU
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x <= 0, 0, 1)

# Identity
def identity(x):
    return x

def identity_derivative(x):
    return np.ones_like(x)

def get_activation(activation: str):
    if activation == "sigmoid":
        return sigmoid, sigmoid_derivative
    elif activation == "relu":
        return relu, relu_derivative
    else:
        return identity, identity_derivative


class Layer:
    def __init__(self, units: int, input_size: int, activation_name: str = "identity"):
        self.W = np.random.rand(units, input_size)                  # (neurons_l, inputs_l)
        self.b = np.random.rand(units, 1)                           # (neurons_l, 1)
        self.activation, self.activationDerivative = get_activation(activation_name)

    def forward(self, inputs):                                      # (samples, inputs_l)
        self.z = np.dot(inputs, self.W.T) + self.b.T                # (samples, neurons_l) + (1, neurons_l) -> (samples, neurons_l)
        self.a = self.activation(self.z)                            # (samples, neurons_l)
        return self.a                                               # (samples, neurons_l)

    def update_weights(self, delta, activations, lr):
        self.W -= lr * np.dot(delta.T, activations)/delta.shape[0]              # (neurons_l, samples) x (samples, neurons_l-1) -> (neurons_l, inputs_l) where neurons_l-1 = inputs_l
        self.b -= lr * (np.sum(delta, axis=0, keepdims=True)/delta.shape[0]).T  # (samples, neurons) -> (1, neurons) -> (neurons, 1)


class NeuralNetwork:
    def __init__(self, layers: list):
        self.layers = layers

    def predict(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)                                             # (samples, inputs) -> (samples, neurons)
        return inputs                                                                  # (samples, neurons_last)

    def fit(self, inputs: np.ndarray, outputs: np.ndarray, epochs: int, lr: float = 0.001, batch_size: int = 32):    
        # input, outputs                                                               (samples, inputs), (samples, )                                                                      
        epoch_losses = []
        for epoch in range(epochs):
            batch_losses = np.array([])
            
            batchCounter = 0
            start = batchCounter * batch_size
            end = end = min(start + batch_size, len(inputs))
            while start < len(inputs):
                inputsBatch = inputs[start:end]                                         # (samples, inputs)
                outputBatch = outputs[start:end].reshape(-1, 1)                         # (samples, 1)
                
                # Forward propagation
                activations = []
                inputsLayer = inputsBatch                                               # (samples, inputs_l)
                activations.append(inputsLayer)
                for layer in self.layers:
                    inputsLayer = layer.forward(inputsLayer)                            # (sample, inputs_l) -> (samples, neurons_l)
                    activations.append(inputsLayer)

                # Backpropagation
                error = activations[-1] - outputBatch                                   # (samples, neurons_L) - (samples, outputs) -> (samples, neurons_L) where neurons_L = outputs
                afdValue = self.layers[-1].activationDerivative(self.layers[-1].z)      # (samples, neurons_L)
                delta = error * afdValue                                                # (samples, neurons_L), neurons_L = neurons_l+1
                deltas = [delta]

                for i in range(len(self.layers) - 2, -1, -1):
                    # The loop starts from the penultimate layer and goes to the first layer.
                    error = np.dot(deltas[0], self.layers[i + 1].W)                     # (samples, neurons_l+1) x (neurons_l+1, inputs_l+1) -> (samples, neurons_l) where inputs_l+1 = neurons_l          
                    afdValue = self.layers[i].activationDerivative(self.layers[i].z)    # (samples, neurons_l) -> (samples, neurons_l)
                    delta = error * afdValue                                            # (samples, neurons_l)
                    deltas.insert(0, delta)
                
                # Epoch loss (MSE)
                batch_losses = np.append(batch_losses, (activations[-1] - outputBatch) ** 2)
         
                # Update weights
                for i in range(len(self.layers)):
                    input_activation = activations[i]
                    output_delta = deltas[i]
                    self.layers[i].update_weights(output_delta, input_activation, lr)

                # Next batch
                batchCounter += 1
                start = batchCounter * batch_size
                end = min(start + batch_size, len(inputs))
            epoch_loss = np.mean(batch_losses)
            epoch_losses.append(epoch_loss)        
            print(f"Epoch {epoch+1}/{epochs}, loss: {epoch_loss}")
        return epoch_losses

if __name__ == "__main__":
    inputs, outputs = generate_data(100)
    split_index = int(0.8 * len(inputs))  

    # Train sets
    train_inputs = inputs[:split_index]
    train_outputs = outputs[:split_index]

    # Test sets
    test_inputs = inputs[split_index:]
    test_outputs = outputs[split_index:]
    
    network = NeuralNetwork([
        Layer(units=4, input_size=3, activation_name="relu"),
        Layer(units=1, input_size=4, activation_name="identity")
    ])
    history = network.fit(train_inputs, train_outputs, epochs=10, lr=0.001, batch_size=8)
    print("Training completed.")

    predictions = network.predict(test_inputs)
    print("Predictions vs True values:")
    for pred, true in zip(predictions.flatten(), test_outputs.flatten()):
        print(f"Pred: {pred:.3f}, True: {true:.3f}")

    final_loss = np.mean((predictions - test_outputs.reshape(-1,1)) ** 2)
    print(f"Final traning loss: {final_loss}")
    
    import matplotlib.pyplot as plt
    plt.plot(history)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()
