import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, inputs=3, outputs=2) -> None:
        self.layers = [inputs, outputs]

        self.activations, self.bias, self.bias_derivative = [], [], []
        for layer in self.layers:
            self.activations.append(np.zeros((1, layer)))

        self.weights = np.random.rand(self.layers[0], self.layers[1])
        self.derivatives = np.zeros((self.layers[0], self.layers[1]))

        self.bias = np.zeros((1, self.layers[-1]))
        self.bias_derivative = np.zeros((1, self.layers[-1]))

    def __propagate_forward(self, inputs):
        self.activations[0] = np.array(inputs).reshape(1, -1)
        self.activations[1] = (self.activations[0] @ self.weights) + self.bias

    def __propagate_backward(self, loss):
        self.derivatives = self.activations[0].T @ loss
        self.bias_derivative = loss * self.bias

    def train(self, dataset, labels, epochs=100, learning_rate=0.001):
        errors = []
        # Loop epoch times
        for _ in range(epochs):
            loss_sum = 0
            for data, expected in zip(dataset, labels):
                expected = np.array(expected)

                # Make a prediction
                self.__propagate_forward(data)

                # Calculating the error a.k.a the loss
                loss = self.activations[-1] - expected

                loss_sum += np.average(loss ** 2)

                # Set the derivatives dL_dW(the gradient)
                self.__propagate_backward(loss)

                # Update the weights in such a way it minimizes the loss function
                self.weights -= learning_rate * self.derivatives

                self.bias -= learning_rate * self.bias_derivative

            errors.append(loss_sum / len(dataset))
        return errors

    def predict(self, input):
        if len(input) != self.layers[0]:
            raise Exception('Invalid input')

        # Make the prediction
        self.__propagate_forward(input)

        return self.activations[-1]

if __name__ == '__main__':
    table = [[0, 0], [1, 0], [0, 1], [1, 1]]
    xor_truth_value = [0, 1, 1, 0]
    and_truth_value = [0, 0, 0, 1]
    or_truth_value = [0, 1, 1, 1]

    and_gate = Perceptron(inputs=2, outputs=1)
    history = and_gate.train(table, and_truth_value, epochs=1000, learning_rate=0.001)

    plt.figure(figsize=(12, 8))
    plot1 = plt.subplot2grid((2, 2), (0, 0), colspan=3, rowspan=1)
    plot2 = plt.subplot2grid((2, 2), (1, 0), colspan=1)
    plot3 = plt.subplot2grid((2, 2), (1, 1), colspan=1)

    plot1.plot([i for i in range(1, 1001)], history)
    plot1.set_title('XOR-gate model error over epoch')
    plot1.set_xlabel('Epoch')
    plot1.set_ylabel('error')

    print(and_gate.predict([1, 0]) > 0.5)
    print(and_gate.predict([0, 1]) > 0.5)
    print(and_gate.predict([0, 0]) > 0.5)
    print(and_gate.predict([1, 1]) > 0.5)
    print()

    or_gate = Perceptron(inputs=2, outputs=1)
    history = or_gate.train(table, or_truth_value, epochs=1000, learning_rate=0.001)

    plot2.plot([i for i in range(1, 1001)], history)
    plot2.set_title('XOR-gate model error over epoch')
    plot2.set_xlabel('Epoch')
    plot2.set_ylabel('error')

    print(or_gate.predict([1, 0]) > 0.5)
    print(or_gate.predict([0, 1]) > 0.5)
    print(or_gate.predict([0, 0]) > 0.5)
    print(or_gate.predict([1, 1]) > 0.5)
    print()

    xor_gate = Perceptron(inputs=2, outputs=1)
    history = xor_gate.train(table, xor_truth_value, epochs=1000, learning_rate=0.001)

    plot3.plot([i for i in range(1, 1001)], history)
    plot3.set_title('XOR-gate model error over epoch')
    plot3.set_xlabel('Epoch')
    plot3.set_ylabel('error')

    print(xor_gate.predict([1, 0]) > 0.5)
    print(xor_gate.predict([0, 1]) > 0.5)
    print(xor_gate.predict([0, 0]) > 0.5)
    print(xor_gate.predict([1, 1]) > 0.5)
    print()
    plt.show()

