using System.Text;

Dictionary<string, double> vals = init(Globals.BITS_PER_PARAMETER, Globals.LOWER_BOUND, Globals.UPPER_BOUND);

List<Specimen> population = [];

for (int i = 0; i < Globals.POPULATION_SIZE; i++)
{
    population.Add(new Specimen());
}

Specimen best = getBestSpecimen(population);
Console.WriteLine("generacja 1 najlepsze przystosowanie: {0}, średnie: {1}",
                  fitness(best),
                  averageFitness(population));

for (int i = 1; i < Globals.GENERATION_COUNT; i++)
{
    List<Specimen> newPopulation = [];


    for (int j = 0; j < Globals.POPULATION_SIZE - 1; j++)
    {
        Specimen parent = tournamentSelection(population);
        Specimen offspring = new Specimen(parent).Mutate();

        newPopulation.Add(offspring);
    }

    Specimen bestSpecimen = getBestSpecimen(population);
    newPopulation.Add(new Specimen(bestSpecimen));
    population = newPopulation;

    Console.WriteLine("generacja {0} najlepsze przystosowanie: {1}, średnie: {2}",
                      i + 1,
                      fitness(bestSpecimen),
                      averageFitness(population));
}

double averageFitness(List<Specimen> specimens)
{
    double avg = 0;

    foreach (var specimen in specimens)
    {
        avg += fitness(specimen);
    }

    return avg / specimens.Count;
}

double fitness(Specimen specimen)
{
    double x1 = vals[specimen.GetParameter(0)];
    double x2 = vals[specimen.GetParameter(1)];
    return Math.Sin(x1 * 0.05) + Math.Sin(x2 * 0.05) + 0.4 * Math.Sin(x1 * 0.15) * Math.Sin(x2 * 0.15);
}

Specimen tournamentSelection(List<Specimen> specimens)
{
    List<Specimen> tempSpecimen = new();

    for (int i = 0; i < Globals.TOURNAMENT_SELECTION_COUNT;)
    {
        Specimen randomSpec = specimens[Globals.rand.Next(Globals.POPULATION_SIZE)];

        bool unique = true;
        foreach (Specimen specimen in tempSpecimen)
        {
            if (specimen.Bits == randomSpec.Bits)
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

    Specimen bestSpecimen = tempSpecimen[0];
    foreach (Specimen specimen in tempSpecimen)
    {
        double fit = fitness(specimen);

        if (fit > fitness(bestSpecimen))
        {
            bestSpecimen = specimen;
        }
    }

    return bestSpecimen;
}

Specimen getBestSpecimen(List<Specimen> specimens)
{
    Specimen bestSpecimen = specimens[0];
    foreach (Specimen specimen in specimens)
    {
        double fit = fitness(specimen);

        if (fit > fitness(bestSpecimen))
        {
            bestSpecimen = specimen;
        }
    }

    return bestSpecimen;
}

static Dictionary<string, double> init(int bits, double min, double max)
{
    Dictionary<string, double> ret = [];

    double step = (max - min) / (Math.Pow(2, bits) - 1);
    double range = min;

    int i;

    for (i = 0; i < Math.Pow(2, bits) - 1; ++i, range += step)
    {
        ret.Add(Convert.ToString(i, 2).PadLeft(bits, '0'), range);
    }

    ret.Add(Convert.ToString(i, 2), max);

    return ret;
}
public static class Globals
{
    public const int POPULATION_SIZE = 21;
    public const int BITS_PER_PARAMETER = 4;
    public const int PARAMETER_COUNT = 2;
    public const int TOURNAMENT_SELECTION_COUNT = 4;
    public const int GENERATION_COUNT = 100;
    public const double LOWER_BOUND = 0.0;
    public const double UPPER_BOUND = 100.0;
    public const double MUTATION_RATE = 0.15;
    public static Random rand = new Random();
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
    }

    public Specimen(Specimen specimen)
    {
        Bits = specimen.Bits;
    }

    public Specimen Mutate()
    {
        StringBuilder sb = new StringBuilder(Bits);

        for (int i = 0; i < sb.Length; ++i)
        {
            if (Globals.rand.NextDouble() < Globals.MUTATION_RATE)
            {
                if (sb[i] == '1')
                {
                    sb[i] = '0';
                }
                else
                {
                    sb[i] = '1';
                }
            }
        }

        Bits = sb.ToString();
        return this;
    }

    public string GetParameter(int parameter)
    {
        return Bits.Substring(parameter * Globals.BITS_PER_PARAMETER, Globals.BITS_PER_PARAMETER);
    }
    public string Bits { get; set; }
}