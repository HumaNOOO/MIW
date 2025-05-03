using System.Collections.Immutable;
using System.Text;

List<Specimen> population = [];

for (int i = 0; i < Globals.POPULATION_SIZE; i++)
{
    population.Add(new Specimen());
}

Specimen best = getBestSpecimen(population);
Console.WriteLine("generacja 1 najlepsze przystosowanie: {0}, średnie: {1} ", best.Fitness(), averageFitness(population));

for (int gen = 1; gen < Globals.GENERATION_COUNT; gen++)
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
    Console.WriteLine("generacja {0} najlepsze przystosowanie: {1}, średnie: {2}", gen+1, getBestSpecimen(newPopulation).Fitness(), averageFitness(population));
}

Console.Write("best params found: ");
getBestSpecimen(population).PrintParams();

double averageFitness(List<Specimen> specimens)
{
    double avg = 0;

    foreach (var specimen in specimens)
    {
        avg += specimen.Fitness();
    }

    return avg / specimens.Count;
}

Specimen tournamentSelection(List<Specimen> specimens)
{
    List<Specimen> tempSpecimen = [];

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
            ++i;
            tempSpecimen.Add(randomSpec);
        }
    }

    return getBestSpecimen(tempSpecimen);
}

Specimen getBestSpecimen(List<Specimen> specimens)
{
    Specimen bestSpecimen = specimens[0];
    foreach (Specimen specimen in specimens)
    {
        double fit = specimen.Fitness();

        if (fit < bestSpecimen.Fitness()) // najlepszy osobnik, czyli posiadający NAJMNIEJSZY bład
        {
            bestSpecimen = specimen;
        }
    }

    return new(bestSpecimen);
}

static Tuple<Specimen, Specimen> crossover(Specimen s1, Specimen s2)
{
    Specimen newSpecimen1 = new();
    Specimen newSpecimen2 = new();

    newSpecimen1.Bits = s1.Bits[..Globals.CROSSOVER_POINT];
    newSpecimen1.Bits += s2.Bits[Globals.CROSSOVER_POINT..];

    newSpecimen2.Bits = s2.Bits[..Globals.CROSSOVER_POINT];
    newSpecimen2.Bits += s1.Bits[Globals.CROSSOVER_POINT..];

    newSpecimen1.RedecodeBits();
    newSpecimen2.RedecodeBits();

    return Tuple.Create(newSpecimen1, newSpecimen2);
}

public static class Globals
{
    public const int POPULATION_SIZE = 13;
    public const int BITS_PER_PARAMETER = 8;
    public const int PARAMETER_COUNT = 3;
    public const int TOURNAMENT_SELECTION_COUNT = 3;
    public const int GENERATION_COUNT = 1000;
    public const double LOWER_BOUND = 0.0;
    public const double UPPER_BOUND = 3.0;
    public const double MUTATION_RATE = 0.20;
    public const int CROSSOVER_POINT = 3;
    public readonly static Random rand = new();
    public readonly static ImmutableList<Tuple<double, double>> NUMS =
    [
        Tuple.Create(-1.0, 0.59554),
        Tuple.Create(-0.8, 0.58813),
        Tuple.Create(-0.6, 0.64181),
        Tuple.Create(-0.4, 0.68587),
        Tuple.Create(-0.2, 0.44783),
        Tuple.Create(0.0, 0.40836),
        Tuple.Create(0.2, 0.38241),
        Tuple.Create(0.4, -0.05933),
        Tuple.Create(0.6, -0.12478),
        Tuple.Create(0.8, -0.36847),
        Tuple.Create(1.0, -0.39935),
        Tuple.Create(1.2, -0.50881),
        Tuple.Create(1.4, -0.63435),
        Tuple.Create(1.6, -0.59979),
        Tuple.Create(1.8, -0.64107),
        Tuple.Create(2.0, -0.51808),
        Tuple.Create(2.2, -0.38127),
        Tuple.Create(2.4, -0.12349),
        Tuple.Create(2.6, -0.09624),
        Tuple.Create(2.8, 0.27893),
        Tuple.Create(3.0, 0.48965),
        Tuple.Create(3.2, 0.33089),
        Tuple.Create(3.4, 0.70615),
        Tuple.Create(3.6, 0.53342),
        Tuple.Create(3.8, 0.43321),
        Tuple.Create(4.0, 0.64790),
        Tuple.Create(4.2, 0.48834),
        Tuple.Create(4.4, 0.18440),
        Tuple.Create(4.6, -0.02389),
        Tuple.Create(4.8, -0.10261),
        Tuple.Create(5.0, -0.33594),
        Tuple.Create(5.2, -0.35101),
        Tuple.Create(5.4, -0.62027),
        Tuple.Create(5.6, -0.55719),
        Tuple.Create(5.8, -0.66377),
        Tuple.Create(6.0, -0.62740)
    ];
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

    public Specimen(Specimen specimen)
    {
        Bits = specimen.Bits;
        RedecodeBits();
    }

    static Specimen()
    {
        Vals = Globals.Init().ToImmutableDictionary();
    }

    public void PrintParams()
    {
        Console.WriteLine("pa: {0}, pb: {1}, pc: {2}", Vals[GetParameter(0)], Vals[GetParameter(1)], Vals[GetParameter(2)]);
    }

    public Specimen Mutate()
    {
        StringBuilder sb = new(Bits);

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

    public void RedecodeBits()
    {
        Parameters = [
            Vals[GetParameter(0)],
            Vals[GetParameter(1)],
            Vals[GetParameter(2)]
        ];
    }

    private string GetParameter(int parameter)
    {
        return Bits.Substring(parameter * Globals.BITS_PER_PARAMETER, Globals.BITS_PER_PARAMETER);
    }

    public double GetParameterValue(int parameter)
    {
        return Parameters[parameter];
    }

    public double Fitness()
    {
        double pa = GetParameterValue(0);
        double pb = GetParameterValue(1);
        double pc = GetParameterValue(2);

        double errorSum = 0;

        foreach (var (x, y) in Globals.NUMS)
        {
            double prediction = pa * Math.Sin(pb * x + pc);
            double error = prediction - y;
            errorSum += Math.Pow(error, 2);
        }

        return errorSum;
    }

    List<double> Parameters;
    public string Bits;
    private static readonly ImmutableDictionary<string, double> Vals;
}