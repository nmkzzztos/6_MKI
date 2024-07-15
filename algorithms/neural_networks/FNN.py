import numpy as np


class FNN:
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        learning_rate: float = 0.01,
        epochs: int = 1000,
        activation: str = "sigmoid",
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.learning_rate = learning_rate
        self.epochs = epochs

        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))

        if activation == "sigmoid":
            self.activation = lambda x: 1 / (1 + np.exp(-x))
            self.activation_derivative = lambda x: x * (1 - x)
        elif activation == "relu":
            self.activation = lambda x: np.maximum(0, x)
            self.activation_derivative = lambda x: np.where(x >= 0, 1, 0)
        elif activation == "softmax":
            self.activation = lambda x: np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
            self.activation_derivative = lambda x: x * (1 - x)
        else:
            raise ValueError(
                "Activation function not supported. Please use 'sigmoid', 'relu', or 'softmax'."
            )

    def _forward_pass(self, X: np.ndarray) -> tuple:
        hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        hidden_layer_output = self.activation(hidden_layer_input)
        output_layer_input = (
            np.dot(hidden_layer_output, self.weights_hidden_output) + self.bias_output
        )
        predicted_output = self.activation(output_layer_input)
        return hidden_layer_output, predicted_output

    def _backward_pass(
        self,
        X: np.ndarray,
        y: np.ndarray,
        hidden_layer_output: np.ndarray,
        predicted_output: np.ndarray,
    ) -> None:
        output_error = y - predicted_output
        output_delta = output_error * self.activation_derivative(predicted_output)

        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.activation_derivative(hidden_layer_output)

        # Update weights and biases
        self.weights_hidden_output += self.learning_rate * np.dot(
            hidden_layer_output.T, output_delta
        )
        self.bias_output += self.learning_rate * np.sum(
            output_delta, axis=0, keepdims=True
        )
        self.weights_input_hidden += self.learning_rate * np.dot(X.T, hidden_delta)
        self.bias_hidden += self.learning_rate * np.sum(
            hidden_delta, axis=0, keepdims=True
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        for _ in range(self.epochs):
            hidden_layer_output, predicted_output = self._forward_pass(X)
            self._backward_pass(X, y, hidden_layer_output, predicted_output)

    def predict(self, X: np.ndarray) -> np.ndarray:
        _, predicted_output = self._forward_pass(X)
        return predicted_output

    def evaluate(self, X: np.ndarray, y: np.ndarray, metric: str = "mse") -> float:
        predictions = self.predict(X)
        if metric == "mse":
            return np.mean((predictions - y) ** 2)
        elif metric == "mae":
            return np.mean(np.abs(predictions - y))
        elif metric == "r2_score":
            return 1 - (np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2))
        elif metric == "accuracy":
            y_pred = np.argmax(predictions, axis=1)
            y_true = np.argmax(y, axis=1)
            return np.mean(y_pred == y_true)
        else:
            raise ValueError("Invalid metric")

    def __str__(self) -> str:
        return f"FNN: input_size={self.input_size}, hidden_size={self.hidden_size}, output_size={self.output_size}, learning_rate={self.learning_rate}, epochs={self.epochs}"

