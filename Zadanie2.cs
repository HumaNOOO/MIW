using System.Collections.Immutable;
using System.Numerics;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;

Zadanie1XOR();
Zadanie2XORPLUSNOR();
Zadanie3SUMATOR();

static void Zadanie1XOR()
{
    Console.WriteLine("######################################## ZADANIE 1 ########################################\n");

    NeuralNetwork nn = new([2, 2, 1], Globals.XOR_LEARNING_RATE);

    int i = 1;
    for (; i <= Globals.XOR_EPOCHS; ++i)
    {
        var data = Globals.XOR_TESTS.ToList();
        data = data.OrderBy(_ => Globals.rand.Next()).ToList();

        bool stopTraining = true;
        double totalError = 0;
        foreach (var (input, output) in data)
        {
            totalError += nn.SetInput([input.Item1, input.Item2]).Forward().Backprop(output).GetMSE(output);
            if (Math.Abs(output[0] - nn.GetResult()[0]) > 0.3)
            {
                stopTraining = false;
            }
        }

        if (i % 1000 == 0 || i == 1)
        {
            Console.WriteLine($"{totalError / Globals.XOR_TESTS.Count}, {i}");
        }

        if (stopTraining) break;
    }

    Console.WriteLine("Trenowanie XOR zajęło {0} epok", i - 1);
    nn.TestXor();
}

static void Zadanie2XORPLUSNOR()
{
    Console.WriteLine("\n\n######################################## ZADANIE 2 ########################################\n");

    NeuralNetwork nn = new([2, 2, 2, 2], Globals.XOR_PLUS_NOR_LEARNING_RATE);

    int i = 1;
    for (; i <= Globals.XOR_PLUS_NOR_EPOCHS; i++)
    {
        var data = Globals.XOR_PLUS_NOR_TESTS.ToList();
        data = data.OrderBy(_ => Globals.rand.Next()).ToList();

        double totalError = 0;
        bool stopTraining = true;
        foreach (var (input, output) in data)
        {
            totalError += nn.SetInput([input.Item1, input.Item2]).Forward().Backprop(output).GetMSE(output);
            if (Math.Abs(output[0] - nn.GetResult()[0]) > 0.4 || Math.Abs(output[1] - nn.GetResult()[1]) > 0.4)
            {
                stopTraining = false;
            }
        }

        if (i % 1000 == 0 || i == 1)
        {
            Console.WriteLine($"{totalError / Globals.XOR_PLUS_NOR_TESTS.Count}, {i}");
        }

        if (stopTraining) break;
    }

    Console.WriteLine("\nTrenowanie XOR + NOR zajęło {0} epok\n", i - 1);
    nn.TestXorPlusNor();
}

static void Zadanie3SUMATOR()
{
    Console.WriteLine("\n\n######################################## ZADANIE 3 ########################################\n");

    NeuralNetwork nn = new([3, 3, 2, 2], Globals.SUMATOR_LEARNING_RATE);

    int i = 1;
    for (; i <= Globals.SUMATOR_EPOCHS; i++)
    {
        var data = Globals.SUMATOR_TESTS.ToList();
        data = data.OrderBy(_ => Globals.rand.Next()).ToList();

        bool stopTraining = true;
        double totalError = 0;
        foreach (var (input, output) in data)
        {
            totalError += nn.SetInput([input.Item1, input.Item2, input.Item3]).Forward().Backprop(output).GetMSE(output);
            if (Math.Abs(output[0] - nn.GetResult()[0]) > 0.4 || Math.Abs(output[1] - nn.GetResult()[1]) > 0.4)
            {
                stopTraining = false;
            }
        }

        if (i % 1000 == 0 || i == 1)
        {
            Console.WriteLine($"{totalError / Globals.SUMATOR_TESTS.Count}, {i}");
        }

        if (stopTraining) break;
    }

    Console.WriteLine("\nTrenowanie sumatora zajęło {0} epok\n", i - 1);
    nn.TestSumator();
}

static class Globals
{
    public static Random rand = new();
    public static readonly Dictionary<(double, double), List<double>> XOR_TESTS = new()
    {
        [(0, 0)] = [0],
        [(0, 1)] = [1],
        [(1, 0)] = [1],
        [(1, 1)] = [0]
    };
    public static readonly Dictionary<(double, double), List<double>> XOR_PLUS_NOR_TESTS = new()
    {
        [(0, 0)] = [0, 1],
        [(0, 1)] = [1, 0],
        [(1, 0)] = [1, 0],
        [(1, 1)] = [0, 0]
    };
    public static readonly Dictionary<(double, double, double), List<double>> SUMATOR_TESTS = new()
    {
        [(0, 0, 0)] = [0, 0],
        [(0, 0, 1)] = [1, 0],
        [(0, 1, 0)] = [1, 0],
        [(0, 1, 1)] = [0, 1],
        [(1, 0, 0)] = [1, 0],
        [(1, 0, 1)] = [0, 1],
        [(1, 1, 0)] = [0, 1],
        [(1, 1, 1)] = [1, 1],
    };
    public static double XOR_EPOCHS = 10000; // w większości przypadków starcza ale dla złych wylosowanych wag i biasów może zając kilka mln epok
    public static double XOR_LEARNING_RATE = 0.2;
    public static double XOR_PLUS_NOR_EPOCHS = 20000;
    public static double XOR_PLUS_NOR_LEARNING_RATE = 0.15;
    public static double SUMATOR_EPOCHS = 50000;
    public static double SUMATOR_LEARNING_RATE = 0.18;
}

class Neuron
{
    public Neuron(int weightsCount, bool bias = false)
    {
        Weights = [];

        for (int i = 0; i < weightsCount; i++)
        {
            Weights.Add(Globals.rand.NextDouble() * 2 - 1);
        }

        Bias = bias ? Globals.rand.NextDouble() * 2 - 1 : 0;
        Delta = 0;
    }

    public List<double> Weights;
    public double Bias;
    public double Value;
    public double Delta;
    public double WeightedSum;
}

class Layer : List<Neuron> { };

class NeuralNetwork
{
    public NeuralNetwork(List<int> layerSizes, double lr)
    {
        LearningRate = lr;
        Layers = [];
        for (int layerIdx = 0; layerIdx < layerSizes.Count; layerIdx++)
        {
            Layer layer = [];

            for (int j = 0; j < layerSizes[layerIdx]; j++)
            {
                if (layerIdx > 0)
                {
                    layer.Add(new Neuron(layerSizes[layerIdx - 1], true));
                }
                else
                {
                    layer.Add(new Neuron(0));
                }
            }
            Layers.Add(layer);
        }
    }

    public NeuralNetwork Forward()
    {
        for (int layerIdx = 1; layerIdx < Layers.Count; layerIdx++)
        {
            var currentLayer = Layers[layerIdx];
            for (int neuronIdx = 0; neuronIdx < currentLayer.Count; neuronIdx++)
            {
                double sum = 0;
                var prevLayer = Layers[layerIdx - 1];
                for (int prevLayerNeuronIdx = 0; prevLayerNeuronIdx < prevLayer.Count; prevLayerNeuronIdx++)
                {
                    sum += prevLayer[prevLayerNeuronIdx].Value * currentLayer[neuronIdx].Weights[prevLayerNeuronIdx];
                }

                sum += currentLayer[neuronIdx].Bias;
                currentLayer[neuronIdx].WeightedSum = sum;
                currentLayer[neuronIdx].Value = Sigmoid(sum);
            }
        }

        return this;
    }

    public NeuralNetwork Backprop(List<double> targets)
    {
        var outputLayerRef = Layers[^1];

        for (int neuronIdx = 0; neuronIdx < outputLayerRef.Count; neuronIdx++)
        {
            outputLayerRef[neuronIdx].Delta = (outputLayerRef[neuronIdx].Value - targets[neuronIdx]) * SigmoidDerivative(outputLayerRef[neuronIdx].WeightedSum);
        }

        for (int layerIdx = Layers.Count - 2; layerIdx >= 1; layerIdx--)
        {
            var currentLayerRef = Layers[layerIdx];
            var nextLayerRef = Layers[layerIdx + 1];

            for (int neuronIdx = 0; neuronIdx < currentLayerRef.Count; neuronIdx++)
            {
                double sum = 0;
                for (int nextNeuronIdx = 0; nextNeuronIdx < nextLayerRef.Count; nextNeuronIdx++)
                {
                    sum += nextLayerRef[nextNeuronIdx].Delta * nextLayerRef[nextNeuronIdx].Weights[neuronIdx];
                }

                currentLayerRef[neuronIdx].Delta = sum * SigmoidDerivative(currentLayerRef[neuronIdx].WeightedSum);
            }
        }

        for (int layerIdx = 1; layerIdx < Layers.Count; layerIdx++)
        {
            var currentLayerRef = Layers[layerIdx];
            var previousLayerRef = Layers[layerIdx - 1];

            foreach (Neuron neuron in currentLayerRef)
            {
                for (int weightIdx = 0; weightIdx < neuron.Weights.Count; weightIdx++)
                {
                    neuron.Weights[weightIdx] -= LearningRate * neuron.Delta * previousLayerRef[weightIdx].Value;
                }

                neuron.Bias -= LearningRate * neuron.Delta;
            }
        }

        return this;
    }

    private static double SigmoidDerivative(double input)
    {
        double s = Sigmoid(input);
        return s * (1 - s);
    }

    public double GetMSE(List<double> expected)
    {
        if (expected.Count != Layers[^1].Count)
        {
            throw new Exception("długości list się nie zgadzają");
        }

        double error = 0;

        for (int outputIdx = 0; outputIdx < Layers[^1].Count; outputIdx++)
        {
            error += Math.Pow(Layers[^1][outputIdx].Value - expected[outputIdx], 2);
        }

        return error / expected.Count;
    }

    public NeuralNetwork SetInput(List<double> values)
    {
        if (values.Count != Layers[0].Count)
        {
            throw new Exception("długości list się nie zgadzają");
        }

        for (int i = 0; i < values.Count; i++)
        {
            Layers[0][i].Value = values[i];
        }

        return this;
    }

    private static double Sigmoid(double result)
    {
        return 1.0 / (1.0 + Math.Exp(-result));
    }

    public List<double> GetResult()
    {
        List<double> results = [];

        foreach (Neuron neuron in Layers[^1])
        {
            results.Add(neuron.Value);
        }

        return results;
    }

    public void TestXor()
    {
        foreach (var (input, output) in Globals.XOR_TESTS)
        {
            Console.WriteLine($"Wejście: {input.Item1} {input.Item2}");
            SetInput([input.Item1, input.Item2]).Forward();
            Console.WriteLine($"Wynik (oczekiwane/rzeczywiste): ({output[0]}/{GetResult()[0]})");
        }
    }

    public void TestXorPlusNor()
    {
        foreach (var (input, output) in Globals.XOR_PLUS_NOR_TESTS)
        {
            Console.WriteLine($"Wejście: {input.Item1} {input.Item2}");
            SetInput([input.Item1, input.Item2]).Forward();
            Console.WriteLine($"Wynik (oczekiwane/rzeczywiste): ({output[0]}/{GetResult()[0]}), ({output[1]}/{GetResult()[1]})");
        }
    }

    public void TestSumator()
    {
        foreach (var (input, output) in Globals.SUMATOR_TESTS)
        {
            Console.WriteLine($"Wejście: {input.Item1} {input.Item2} {input.Item3}");
            SetInput([input.Item1, input.Item2, input.Item3]).Forward();
            Console.WriteLine($"Wynik (oczekiwane/rzeczywiste): ({output[0]}/{GetResult()[0]}), ({output[1]}/{GetResult()[1]})");
        }
    }

    private readonly double LearningRate;
    public List<Layer> Layers;
}