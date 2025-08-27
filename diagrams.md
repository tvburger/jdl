# Neural Network Implementation Diagrams

This document contains detailed class and sequence diagrams for each module of the neural network implementation.

## Core Model Module

### Neural Network Core Components
```mermaid
classDiagram
    class NeuralNetwork {
        <<interface>>
        +getArchitecture()
        +getWidth(int)
        +getDepth()
        +getNeuron(int, int)
        +getOutputConnections(int, int)
        +getParameterCount()
    }

    class DefaultNeuralNetwork {
        -layers: List<List<Neuron>>
        +DefaultNeuralNetwork(List<List<Neuron>>)
    }

    class Architecture {
        +getWidth(int)
        +getDepth()
        +validate()
    }

    class Neuron {
        <<interface>>
        +getOutput()
        +findWeight(Neuron)
        +computeOutput()
        +updateWeights()
    }

    class InputNeuron {
        -value: float
        +setValue(float)
        +getOutput()
    }

    class ActivationFunction {
        <<interface>>
        +activate(float)
        +derivative(float)
    }

    NeuralNetwork <|.. DefaultNeuralNetwork
    DefaultNeuralNetwork o-- Architecture
    DefaultNeuralNetwork o-- Neuron
    Neuron <|.. InputNeuron
    Neuron --> ActivationFunction
```

### Activation Functions
```mermaid
classDiagram
    class ActivationFunction {
        <<interface>>
        +activate(float)
        +derivative(float)
    }

    class ReLU {
        +activate(float)
        +derivative(float)
    }

    class Sigmoid {
        +activate(float)
        +derivative(float)
    }

    class Linear {
        +activate(float)
        +derivative(float)
    }

    class Step {
        +activate(float)
        +derivative(float)
    }

    ActivationFunction <|.. ReLU
    ActivationFunction <|.. Sigmoid
    ActivationFunction <|.. Linear
    ActivationFunction <|.. Step
```

### Training Components
```mermaid
classDiagram
    class Trainer {
        <<interface>>
        +train(DataSet)
    }

    class GradientDescent {
        -network: NeuralNetwork
        -lossFunction: LossFunction
        -learningRate: float
        +train(DataSet)
        -backpropagate(float[])
    }

    class LossFunction {
        <<interface>>
        +calculate(float[], float[])
        +derivative(float[], float[])
    }

    class MSE {
        +calculate(float[], float[])
        +derivative(float[], float[])
    }

    class DataSet {
        -inputs: float[][]
        -targets: float[][]
        +getSize()
        +getInput(int)
        +getTarget(int)
    }

    Trainer <|.. GradientDescent
    GradientDescent --> LossFunction
    LossFunction <|.. MSE
    GradientDescent --> DataSet
```

## Multi-Layer Perceptron Module

```mermaid
classDiagram
    class MultiLayerPerceptron {
        -layers: List<Layer>
        -activationFunction: ActivationFunction
        +forward(float[])
        +backward(float[])
        +updateWeights(float)
    }

    class Layer {
        -neurons: List<Neuron>
        -activation: ActivationFunction
        +compute()
        +updateWeights()
        +getOutput()
    }

    MultiLayerPerceptron *-- Layer
    Layer *-- Neuron
    Layer --> ActivationFunction
```

## Perceptron Module

```mermaid
classDiagram
    class Perceptron {
        -weights: float[]
        -bias: float
        -learningRate: float
        +train(float[], float)
        +predict(float[])
        +updateWeights(float[], float)
    }

    class AssociationUnit {
        -weights: Map<String, Float>
        -perceptron: Perceptron
        +associate(String, float[])
        +recall(float[])
        +getAssociations()
    }

    class PerceptronTrainer {
        -perceptron: Perceptron
        -learningRate: float
        +train(DataSet)
        +setLearningRate(float)
    }

    class DataSets {
        +createORDataSet()
        +createANDDataSet()
        +createXORDataSet()
    }

    Perceptron --> PerceptronTrainer
    AssociationUnit --> Perceptron
    PerceptronTrainer --> DataSets
```

## Functions Module

```mermaid
classDiagram
    class Estimator {
        <<interface>>
        +estimate(float[])
        +train(DataSet)
    }

    class AddEstimator {
        -network: NeuralNetwork
        -trainer: Trainer
        +estimate(float[])
        +train(DataSet)
    }

    class MultiplicationEstimator {
        -network: NeuralNetwork
        -trainer: Trainer
        +estimate(float[])
        +train(DataSet)
    }

    class XorEstimator {
        -network: NeuralNetwork
        -trainer: Trainer
        +estimate(float[])
        +train(DataSet)
    }

    Estimator <|.. AddEstimator
    Estimator <|.. MultiplicationEstimator
    Estimator <|.. XorEstimator
```

## Sequence Diagrams

### Training Process
```mermaid
sequenceDiagram
    participant Client
    participant Trainer
    participant NeuralNetwork
    participant Neuron
    participant LossFunction
    
    Client->>Trainer: train(dataset)
    loop For each epoch
        loop For each sample
            Trainer->>NeuralNetwork: forward(input)
            loop For each layer
                NeuralNetwork->>Neuron: computeOutput()
                Neuron-->>NeuralNetwork: output
            end
            NeuralNetwork-->>Trainer: output
            Trainer->>LossFunction: calculate(output, target)
            LossFunction-->>Trainer: loss
            Trainer->>NeuralNetwork: backward(gradient)
            loop For each layer
                NeuralNetwork->>Neuron: updateWeights()
            end
        end
    end
    Trainer-->>Client: trained network
```

### Prediction/Inference Process
```mermaid
sequenceDiagram
    participant Client
    participant NeuralNetwork
    participant Layer
    participant Neuron
    participant ActivationFunction
    
    Client->>NeuralNetwork: predict(input)
    loop For each layer
        NeuralNetwork->>Layer: forward()
        loop For each neuron
            Layer->>Neuron: computeOutput()
            Neuron->>ActivationFunction: activate(sum)
            ActivationFunction-->>Neuron: activated value
            Neuron-->>Layer: output
        end
        Layer-->>NeuralNetwork: layer output
    end
    NeuralNetwork-->>Client: prediction
```

### Weight Update Process
```mermaid
sequenceDiagram
    participant GradientDescent
    participant NeuralNetwork
    participant Neuron
    
    GradientDescent->>NeuralNetwork: backpropagate(error)
    loop For each layer backwards
        NeuralNetwork->>Neuron: computeGradient()
        Neuron-->>NeuralNetwork: gradient
        NeuralNetwork->>Neuron: updateWeights(learningRate)
        Neuron-->>NeuralNetwork: updated
    end
    NeuralNetwork-->>GradientDescent: completed
```
