using System.Collections.Immutable;
using System.Diagnostics;
using System.Text;

NeuralNetwork nn = new([2, 2, 1]);

List<Specimen> population = [];

for (int i = 0; i < Globals.POPULATION_SIZE; i++)
{
    population.Add(new Specimen());
}

Specimen best = getBestSpecimen(population);
Console.WriteLine("generacja 0 najlepsze przystosowanie: {0}, średnie przystosowanie: {1} ",
                  fitness(best, nn),
                  averageFitness(population, nn));

for (int gen = 1; gen <= Globals.GENERATION_COUNT; gen++)
{
    List<Specimen> newPopulation = [];

    // 1. 4 osobniki: selekcja + krzyżowanie (2 krzyżowania => 4 potomków)
    Specimen s1 = tournamentSelection(population);
    Specimen s2 = tournamentSelection(population);
    Tuple<Specimen, Specimen> cross1 = crossover(s1, s2);
    newPopulation.Add(cross1.Item1);
    newPopulation.Add(cross1.Item2);

    s1 = tournamentSelection(population);
    s2 = tournamentSelection(population);
    Tuple<Specimen, Specimen> cross2 = crossover(s1, s2);
    newPopulation.Add(cross2.Item1);
    newPopulation.Add(cross2.Item2);

    // 2. 4 osobniki: selekcja + mutacja
    for (int i = 0; i < 4; i++)
    {
        Specimen specimen = tournamentSelection(population);
        specimen.Mutate();
        newPopulation.Add(specimen);
    }

    // 3. 4 osobniki: selekcja + krzyżowanie + mutacja
    s1 = tournamentSelection(population);
    s2 = tournamentSelection(population);
    Tuple<Specimen, Specimen> cross3 = crossover(s1, s2);
    cross3.Item1.Mutate();
    cross3.Item2.Mutate();
    newPopulation.Add(cross3.Item1);
    newPopulation.Add(cross3.Item2);

    s1 = tournamentSelection(population);
    s2 = tournamentSelection(population);
    Tuple<Specimen, Specimen> cross4 = crossover(s1, s2);
    cross4.Item1.Mutate();
    cross4.Item2.Mutate();
    newPopulation.Add(cross4.Item1);
    newPopulation.Add(cross4.Item2);

    // 4. Dodaj najlepszego z poprzedniej generacji
    Specimen bestSpecimen = getBestSpecimen(population);
    newPopulation.Add(new Specimen(bestSpecimen));

    population = newPopulation;
    Console.WriteLine("generacja {0} najlepsze przystosowanie: {1}, średnie przystosowanie: {2}",
                      gen,
                      fitness(getBestSpecimen(newPopulation), nn).ToString("F15"),
                      averageFitness(population, nn).ToString("F15"));
}
nn.PrintParams();

double averageFitness(List<Specimen> specimens,
                      NeuralNetwork nn)
{
    double avg = 0;

    foreach (var specimen in specimens)
    {
        avg += fitness(specimen, nn);
    }

    return avg / specimens.Count;
}

double fitness(Specimen specimen,
               NeuralNetwork nn)
{
    List<double> weights = [specimen.GetParameterValue(0),
                            specimen.GetParameterValue(1),
                            specimen.GetParameterValue(2),
                            specimen.GetParameterValue(3),
                            specimen.GetParameterValue(4),
                            specimen.GetParameterValue(5)];

    List<double> biases = [specimen.GetParameterValue(6),
                           specimen.GetParameterValue(7),
                           specimen.GetParameterValue(8)];

    nn.SetWeightsAndBiases(weights, biases);

    double error = 0;

    foreach (var (input, output) in Globals.expected)
    {
        nn.SetValues([input.Item1, input.Item2]);
        error += Math.Pow(output - nn.Forward().GetResult(), 2);
    }

    return error;
}

Specimen tournamentSelection(List<Specimen> specimens)
{
    List<Specimen> tempSpecimen = new();

    for (int i = 0; i < Globals.TOURNAMENT_SELECTION_COUNT;)
    {
        Specimen randomSpec = specimens[Globals.rand.Next(Globals.POPULATION_SIZE)];

        bool unique = true;
        foreach (Specimen spec in tempSpecimen)
        {
            if (spec.Bits == randomSpec.Bits)
            {
                unique = false;
                break;
            }
        }

        if (unique)
        {
            tempSpecimen.Add(randomSpec);
            ++i;
        }
    }

    return getBestSpecimen(tempSpecimen);
}

Specimen getBestSpecimen(List<Specimen> specimens)
{
    Specimen bestSpecimen = specimens[0];
    double bestFitness = fitness(bestSpecimen, nn);

    foreach (Specimen specimen in specimens)
    {
        double fit = fitness(specimen, nn);

        if (fit < bestFitness) // najlepszy osobnik, czyli posiadający NAJMNIEJSZY bład
        {
            bestSpecimen = specimen;
            bestFitness = fit;
        }
    }

    return new(bestSpecimen);
}

static Tuple<Specimen, Specimen> crossover(Specimen s1, Specimen s2)
{
    return Tuple.Create(new Specimen(s1.Bits[..Globals.CROSSOVER_POINT] + s2.Bits[Globals.CROSSOVER_POINT..]),
                        new Specimen(s2.Bits[..Globals.CROSSOVER_POINT] + s1.Bits[Globals.CROSSOVER_POINT..]));
}

public static class Globals
{
    public const int POPULATION_SIZE = 13;
    public const int BITS_PER_PARAMETER = 8;
    public const int PARAMETER_COUNT = 9;
    public const int TOURNAMENT_SELECTION_COUNT = 3;
    public const int GENERATION_COUNT = 1000;
    public const double LOWER_BOUND = -10.0;
    public const double UPPER_BOUND = 10.0;
    public const double MUTATION_RATE = 0.25;
    public const int CROSSOVER_POINT = (BITS_PER_PARAMETER * PARAMETER_COUNT) / 2;
    public readonly static Random rand = new();
    public static Dictionary<string, double> Init()
    {
        Dictionary<string, double> ret = [];

        double step = (UPPER_BOUND - LOWER_BOUND) / (Math.Pow(2, BITS_PER_PARAMETER) - 1);
        double range = LOWER_BOUND;

        int i;

        for (i = 0; i < Math.Pow(2, BITS_PER_PARAMETER) - 1; ++i, range += step)
        {
            ret.Add(Convert.ToString(i, 2).PadLeft(BITS_PER_PARAMETER, '0'), range);
        }

        ret.Add(Convert.ToString(i, 2), UPPER_BOUND);

        return ret;
    }

    public static readonly Dictionary<(double, double), double> expected = new()
    {
        [(0.0, 0.0)] = 0,
        [(1.0, 0.0)] = 1,
        [(0.0, 1.0)] = 1,
        [(1.0, 1.0)] = 0
    };
}

class Specimen
{
    public Specimen()
    {
        StringBuilder sb = new();
        for (int i = 0; i < Globals.BITS_PER_PARAMETER * Globals.PARAMETER_COUNT; ++i)
        {
            sb.Append(Globals.rand.Next(2) == 1 ? '1' : '0');
        }

        Bits = sb.ToString();
        RedecodeBits();
    }

    static Specimen()
    {
        Vals = Globals.Init().ToImmutableDictionary();
    }

    public Specimen(string bits)
    {
        Bits = bits;
        RedecodeBits();
    }

    public Specimen(Specimen specimen)
    {
        Bits = specimen.Bits;
        RedecodeBits();
    }

    public Specimen Mutate()
    {
        StringBuilder sb = new StringBuilder(Bits);

        for (int i = 0; i < sb.Length; ++i)
        {
            if (Globals.rand.NextDouble() < Globals.MUTATION_RATE)
            {
                sb[i] = sb[i] == '1' ? '0' : '1';
            }
        }

        Bits = sb.ToString();
        RedecodeBits();
        return this;
    }

    private string GetParameter(int parameter)
    {
        return Bits.Substring(parameter * Globals.BITS_PER_PARAMETER, Globals.BITS_PER_PARAMETER);
    }

    public void RedecodeBits()
    {
        Parameters = [];
        for (int i = 0; i < Globals.PARAMETER_COUNT; ++i)
        {
            Parameters.Add(Vals[GetParameter(i)]);
        }
    }

    public double GetParameterValue(int parameter)
    {
        return Parameters[parameter];
    }

    private static readonly ImmutableDictionary<string, double> Vals;
    private List<double> Parameters;
    public string Bits;
}

class Neuron
{
    public Neuron(int weightsCount, bool bias = false)
    {
        Weights = [];

        for (int i = 0; i < weightsCount; i++)
        {
            Weights.Add(Globals.rand.NextDouble());
        }

        Bias = bias ? 0 : 1;
    }

    public List<double> Weights;
    public double Bias;
    public double Value;
}

class Layer : List<Neuron> { };

class NeuralNetwork
{
    public NeuralNetwork(List<int> layerSizes)
    {
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

    public void SetWeightsAndBiases(List<double> weights, List<double> biases)
    {
        Layers[1][0].Bias = biases[0];
        Layers[1][1].Bias = biases[1];
        Layers[2][0].Bias = biases[2];

        Layers[1][0].Weights[0] = weights[0];
        Layers[1][0].Weights[1] = weights[1];

        Layers[1][1].Weights[0] = weights[2];
        Layers[1][1].Weights[1] = weights[3];

        Layers[2][0].Weights[0] = weights[4];
        Layers[2][0].Weights[1] = weights[5];
    }

    public void PrintParams()
    {
        Console.WriteLine("\nZnalezione optymalne parametry po {0} generacjach, Wx(y,z) = X waga w Y Neuronie w Z warstwie, B(i,j) = I bias w J warstwie:", Globals.GENERATION_COUNT);

        for (int layerIdx = 1; layerIdx < Layers.Count; layerIdx++)
        {
            for (int neuronIdx = 0; neuronIdx < Layers[layerIdx].Count; neuronIdx++)
            {
                for (int weightIdx = 0; weightIdx < Layers[layerIdx][neuronIdx].Weights.Count; weightIdx++)
                {
                    Console.WriteLine("W{0}({1},{2}) = {3}", weightIdx, neuronIdx, layerIdx, Layers[layerIdx][neuronIdx].Weights[weightIdx]);
                }
            }
        }

        Console.WriteLine();

        for (int layerIdx = 1; layerIdx < Layers.Count; layerIdx++)
        {
            for (int neuronIdx = 0; neuronIdx < Layers[layerIdx].Count; neuronIdx++)
            {
                Console.WriteLine("B({0},{1}) = {2}", neuronIdx, layerIdx, Layers[layerIdx][neuronIdx].Bias);
            }
        }

        Console.WriteLine("\n===== TEST =====");

        foreach (var (input, expected) in Globals.expected)
        {
            SetValues([input.Item1, input.Item2]).Forward();
            Console.WriteLine("Wartości wejściowe: ({0},{1}), wynik: {2}({3}), oczekiwane: {4}, błąd: {5}", input.Item1, input.Item2, Result.ToString("F15"), (Result <= 0.5 ? 0 : 1), expected, Math.Abs(Result - expected).ToString("F15"));
        }
    }

    public NeuralNetwork Forward()
    {
        for (int layerIdx = 1; layerIdx < Layers.Count; layerIdx++)
        {
            var currentLayer = Layers[layerIdx];
            for (int neuronIdx = 0; neuronIdx < currentLayer.Count; neuronIdx++)
            {
                double weightedSum = 0;
                var prevLayer = Layers[layerIdx - 1];
                for (int prevLayerNeuronIdx = 0; prevLayerNeuronIdx < prevLayer.Count; prevLayerNeuronIdx++)
                {
                    weightedSum += prevLayer[prevLayerNeuronIdx].Value * currentLayer[neuronIdx].Weights[prevLayerNeuronIdx];
                }

                weightedSum += currentLayer[neuronIdx].Bias;

                if (layerIdx < Layers.Count - 1)
                {
                    currentLayer[neuronIdx].Value = ReLU(weightedSum);
                }
                else
                {
                    Result = Sigmoid(weightedSum);
                }
            }
        }

        return this;
    }

    public NeuralNetwork SetValues(List<double> values)
    {
        Layers[0][0].Value = values[0];
        Layers[0][1].Value = values[1];

        return this;
    }

    private static double Sigmoid(double result)
    {
        return 1.0 / (1.0 + Math.Exp(-result));
    }

    private static double ReLU(double result)
    {
        return Math.Max(result, 0);
    }

    public double GetResult()
    {
        return Result;
    }

    private double Result;
    public List<Layer> Layers;
}