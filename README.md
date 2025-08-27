# Java Deep Learning (JDL) Framework

A Java-based deep learning framework that implements various neural network architectures and learning algorithms from scratch. This project serves as both a learning tool and a practical implementation of neural network concepts.

For detailed class and sequence diagrams of all modules, please see the [Technical Diagrams](diagrams.md).

## Project Structure

The project is organized into several modules:

### Core Modules

- **model**: Contains the core neural network components
  - Neural network building blocks (`NeuralNetwork`, `Neuron`, `InputNeuron`)
  - Activation functions (ReLU, Sigmoid, Linear, Step)
  - Learning algorithms (Gradient Descent)
  - Dataset handling and utilities
  - Weight initialization strategies

- **mlp**: Implementation of Multi-Layer Perceptron
  - Configurable network depth and layer sizes
  - Flexible activation function selection
  - Support for both hidden and output layers

- **perceptron**: Basic perceptron implementation
  - Simple binary classification
  - Includes sample datasets (OR function, Circles and Lines)
  - Custom initialization and training strategies

### Function Modules

Practical implementations of various neural network applications:

- **add**: Neural network for addition operations
- **multiplication**: Neural network for multiplication operations
- **xor**: Neural network implementation of the XOR function

## Features

- Modular architecture for easy extension and experimentation
- Various activation functions (ReLU, Sigmoid, Linear, Step)
- Customizable neural network architectures
- Gradient descent-based learning
- Built-in loss functions (MSE)
- Data set handling utilities
- Weight initialization strategies

## Prerequisites

- Java (Latest LTS version)
- Maven for dependency management and building

## Building the Project

To build the project, use Maven:

```bash
mvn clean install
```

## Module Dependencies

```
model
├── perceptron
├── mlp
└── functions
    ├── add
    ├── multiplication
    └── xor
```

## Architecture

The framework follows a modular design where:
- `model` provides the core neural network infrastructure
- Each specialized implementation (perceptron, mlp) extends the core components
- Function modules demonstrate practical applications

## Learning Resources

This project includes implementations of:
- Basic Perceptron
- Multi-Layer Perceptron
- Common activation functions
- Gradient Descent optimization
- Various practical neural network applications

Perfect for learning about:
- Neural network fundamentals
- Backpropagation
- Different types of activation functions
- Weight initialization strategies
- Practical neural network applications

## License

[Add appropriate license information]
