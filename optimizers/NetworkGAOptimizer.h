
#ifndef NETWORK_GA_OPTIMIZER_H_
#define NETWORK_GA_OPTIMIZER_H_

#define MAX_GENS 1000
#define PRINT_INTERVAL 10
#define INSTANTIATE_REAL_GENOME true

#include <ga/GARealGenome.h>
#include <ga/ga.h>
#include <ga/std_stream.h>
#include <DynamicalSystemLimiter.h>
#include <vector>
#include <memory>

void printStats(GAGeneticAlgorithm &ga, unsigned int steps)
{
    float best = ga.statistics().current(GAStatistics::Maximum);
    float worst = ga.statistics().current(GAStatistics::Minimum);
    float average = ga.statistics().current(GAStatistics::Mean);
    float stdev = ga.statistics().current(GAStatistics::Deviation);

    std::cout << "[Generation " << ga.generation() << "]\n"
              << "  Best Fitness:    " << best << "\n"
              << "  Average Fitness: " << average << "\n"
              << "  Worst Fitness:   " << worst << "\n"
              << "  Std Dev:         " << stdev << "\n";
}

template <typename TObjective>
class NetworkGAOptimizer
{
public:
    typedef TObjective Objective;
    typedef typename Objective::Network Network;
    typedef typename Network::Neuron Neuron;
    typedef typename Network::Synapsis Synapsis;

    typedef DynamicalSystemLimiter<Neuron> NeuronLimiter;
    typedef DynamicalSystemLimiter<Synapsis> SynapsisLimiter;

    enum parameter
    {
        pConv,
        pRepl,
        popSize,
        pCross,
        pMut,
        nGens,
        nElite,
        n_parameters,
    };

    struct ConstructorArgs
    {
        float params[n_parameters];
    };

    NetworkGAOptimizer(ConstructorArgs &args, Objective &o)
        : storedObjective(o)
    {
        std::copy(std::begin(args.params), std::end(args.params), std::begin(params));
        this->configured = false;
        this->seed = time(NULL);
    }

    // I allow for users to set a seed
    // that ensures the reproducibility of experiments performed using this feature
    NetworkGAOptimizer(ConstructorArgs &args, Objective &o, int seed)
        : storedObjective(o)
    {
        std::copy(std::begin(args.params), std::end(args.params), std::begin(params));
        this->configured = false;
        this->seed = seed;
    }

    size_t add_neuron(NeuronLimiter limits)
    {
        size_t id = neurons.size();
        NeuronBlueprint blueprint;
        blueprint.id = id;
        blueprint.limits = limits;

        neurons.push_back(blueprint);

        return id;
    }

    void add_synapsis(size_t neuron1, size_t neuron2, SynapsisLimiter limits)
    {
        SynapsisBlueprint blueprint;
        blueprint.n1 = neuron1;
        blueprint.n2 = neuron2;
        blueprint.limits = limits;

        synapsises.push_back(blueprint);
    }

    Network generate()
    {
        if (configured == false)
        {
            configure();
        }

        unsigned int steps = 0;
        while (!terminateSimulation(*this->ga))
        {
            this->ga->step();

            if (steps % PRINT_INTERVAL == 0)
            {
                printStats(*this->ga, steps);
            }
            steps++;
        }

        const GAGenome &bestGenome = ga->statistics().bestIndividual();
        return genomeToNetwork(bestGenome);
    }

private:
    Objective storedObjective;
    std::unique_ptr<GASteadyStateGA> ga;
    bool configured;
    int seed;

    float params[n_parameters];

    struct NeuronBlueprint
    {
        size_t id;
        NeuronLimiter limits;
    };
    struct SynapsisBlueprint
    {
        size_t n1;
        size_t n2;
        SynapsisLimiter limits;
    };

    std::vector<NeuronBlueprint> neurons;
    std::vector<SynapsisBlueprint> synapsises;

    void configure()
    {

        GARealGenome *realGenome = new GARealGenome(createGenome());
        ga = std::unique_ptr<GASteadyStateGA>(new GASteadyStateGA(*realGenome));

        ga->parameters(createParameterList(params));
        ga->pReplacement(params[parameter::pRepl]);
        ga->pConvergence(params[parameter::pConv]);
        this->ga->initialize(seed);

        configured = true;
    }

    GAParameterList createParameterList(float *setupParams)
    {
        GAParameterList params;

        params.set(gaNpopulationSize, setupParams[parameter::popSize]);
        params.set(gaNpCrossover, setupParams[parameter::pCross]);
        params.set(gaNpMutation, setupParams[parameter::pMut]);
        params.set(gaNnGenerations, setupParams[parameter::nGens]);
        params.set(gaNelitism, setupParams[parameter::nElite]);

        // avoids unnecessary writes to the console
        params.set(gaNscoreFrequency, 0);
        params.set(gaNflushFrequency, 0);

        params.set(gaNselectScores, (int)GAStatistics::AllScores);

        GASteadyStateGA::registerDefaultParameters(params);

        return params;
    }

    GARealGenome createGenome()
    {
        GARealAlleleSetArray alleles = limitsToAlleleSet();
        GARealGenome genome(alleles, objectiveFunc);
        genome.userData(this);

        genome.initializer(GARealUniformInitializer);
        genome.crossover(GARealOnePointCrossover);
        genome.mutator(GARealGaussianMutator);

        return genome;
    }

    GARealAlleleSetArray limitsToAlleleSet()
    {
        GARealAlleleSetArray alleleArray;

        for (const NeuronBlueprint &blueprint : neurons)
        {
            addLimits<NeuronLimiter>(blueprint.limits, alleleArray);
        }

        for (const SynapsisBlueprint &blueprint : synapsises)
        {
            addLimits<SynapsisLimiter>(blueprint.limits, alleleArray);
        }

        return alleleArray;
    }

    template <typename Limiter>
    void addLimits(Limiter limiter, GARealAlleleSetArray &alleleArray)
    {
        for (int i = 0; i < Limiter::parameter::n_parameters; i++)
        {
            auto param = static_cast<typename Limiter::parameter>(i);
            float min = limiter.getMin(param);
            float max = limiter.getMax(param);
            alleleArray.add(GARealAlleleSet(min, max));
        }

        for (int i = 0; i < Limiter::variable::n_variables; i++)
        {
            auto var = static_cast<typename Limiter::variable>(i);
            float min = limiter.getMin(var);
            float max = limiter.getMax(var);

            alleleArray.add(GARealAlleleSet(min, max));
        }
    }

    Neuron *genomeToNeuron(const GARealGenome &genome, int start)
    {
        typename Neuron::ConstructorArgs args = {};

        // collect parameters from the genome
        for (int i = 0; i < Neuron::n_parameters; ++i)
        {
            args.params[i] = genome.gene(start + i);
        }

        Neuron *neuron = new Neuron(args);

        // collect initial neuron state from the genome
        for (int i = 0; i < Neuron::n_variables; ++i)
        {
            auto var = static_cast<typename Neuron::variable>(i);
            auto value = genome.gene(start + Neuron::n_parameters + i);
            neuron->set(var, value);
        }

        return neuron;
    }

    Network genomeToNetwork(const GAGenome &genome)
    {
        GARealGenome realGenome = static_cast<const GARealGenome &>(genome);
        Network network;

        int neuron_size = Neuron::n_parameters + Neuron::n_variables;
        for (int neuron_idx = 0; neuron_idx < (int)neurons.size(); neuron_idx++)
        {
            Neuron *neuron = genomeToNeuron(realGenome, neuron_idx * neuron_size);
            network.add_neuron(neuron);
        }

        int synapsis_size = Synapsis::n_parameters + Synapsis::n_variables;
        int synapsis_start = neurons.size() * neuron_size;
        for (int synapsis_idx = 0; synapsis_idx < (int)synapsises.size(); synapsis_idx++)
        {
            SynapsisBlueprint blueprint = synapsises[synapsis_idx];
            Neuron *n1 = network.get_neurons()[blueprint.n1];
            Neuron *n2 = network.get_neurons()[blueprint.n2];

            int genome_idx = synapsis_start + synapsis_idx * synapsis_size;

            // this only works for electrical synapsis.
            // There is no common interface for synapsis instanciation
            Synapsis *synapsis = new Synapsis(*n1, Neuron::v, *n2, Neuron::v, realGenome.gene(genome_idx), realGenome.gene(genome_idx + 1));
            network.add_synapsis(synapsis);
        }

        return network;
    }

    static void clean_network(Network &network)
    {
        for (Neuron *n : network.get_neurons())
        {
            delete n;
        }

        for (Synapsis *s : network.get_synapsises())
        {
            delete s;
        }
    }

    static float objectiveFunc(GAGenome &g)
    {
        NetworkGAOptimizer *optimizer = static_cast<NetworkGAOptimizer *>(g.userData());
        Network network = optimizer->genomeToNetwork(g);
        float fitness = optimizer->storedObjective.evaluate(network);
        clean_network(network);

        if (!std::isfinite(fitness))
        {
            return 0.0f;
        }
        else
        {
            return fitness;
        }
    }

    static bool terminateSimulation(GAGeneticAlgorithm &ga)
    {
        const float convNow = ga.convergence();
        const int maxGens = ga.nGenerations();
        const float convGoal = ga.pConvergence();

        return (ga.generation() >= maxGens || convNow >= convGoal);
    }
};

#endif // NEURON_GA_OPTIMIZER_H_
