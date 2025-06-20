
#ifndef NETWORK_GA_OPTIMIZER_H_
#define NETWORK_GA_OPTIMIZER_H_

#define PRINT_INTERVAL 1
// this line is needed to compile the real genomes, if removed the code will not compile
#define INSTANTIATE_REAL_GENOME true

#include <ga/GARealGenome.h>
#include <ga/ga.h>
#include <ga/std_stream.h>
#include <NetworkBlueprint.h>

#include <vector>
#include <memory>

template <typename TObjective>
class NetworkGAOptimizer
{
public:
    typedef TObjective Objective;
    typedef typename Objective::Network Network;
    typedef typename Network::Neuron Neuron;
    typedef typename Network::Synapsis Synapsis;

    typedef NetworkBlueprint<Network> Blueprint;

    enum parameter
    {
        pConv,
        pRepl,
        popSize,
        pCross,
        pMut,
        nGens,
        n_parameters,
    };

    struct ConstructorArgs
    {
        float params[n_parameters];
    };

    NetworkGAOptimizer(ConstructorArgs &args, Objective &o, Blueprint &b)
        : storedObjective(o), blueprint(b), ga(createGenome())
    {
        std::copy(std::begin(args.params), std::end(args.params), std::begin(params));

        ga.parameters(createParameterList(params));
        ga.pReplacement(params[parameter::pRepl]);
        ga.pConvergence(params[parameter::pConv]);
        this->ga.initialize(time(NULL));
    }

    // I allow for users to set a seed
    // that ensures the reproducibility of experiments performed using this feature
    NetworkGAOptimizer(ConstructorArgs &args, Objective &o, Blueprint &b, int seed)
        : storedObjective(o), blueprint(b), ga(createGenome())
    {
        std::copy(std::begin(args.params), std::end(args.params), std::begin(params));

        ga.parameters(createParameterList(params));
        ga.pReplacement(params[parameter::pRepl]);
        ga.pConvergence(params[parameter::pConv]);
        this->ga.initialize(seed);
    }

    Network generate()
    {
        unsigned int steps = 0;
        while (!terminateSimulation(this->ga))
        {
            this->ga.step();

            if (steps % PRINT_INTERVAL == 0)
            {
                printStats(this->ga, steps);
            }
            steps++;
        }

        const GAGenome &bestGenome = ga.statistics().bestIndividual();
        return blueprint.genome_to_network(bestGenome);
    }

private:
    Objective storedObjective;
    Blueprint blueprint;
    GASteadyStateGA ga;

    float params[n_parameters];

    GARealGenome createGenome()
    {
        GARealAlleleSetArray alleles = this->blueprint.generate_alleles();
        GARealGenome genome(alleles, objectiveFunc);
        genome.userData(this);

        genome.initializer(GARealUniformInitializer);
        genome.crossover(GARealOnePointCrossover);
        genome.mutator(GARealGaussianMutator);

        return genome;
    }

    static GAParameterList createParameterList(float *setupParams)
    {
        GAParameterList params;

        GASteadyStateGA::registerDefaultParameters(params);

        params.set(gaNpopulationSize, setupParams[parameter::popSize]);
        params.set(gaNpCrossover, setupParams[parameter::pCross]);
        params.set(gaNpMutation, setupParams[parameter::pMut]);
        params.set(gaNnGenerations, setupParams[parameter::nGens]);

        // avoids unnecessary writes to the console
        params.set(gaNscoreFrequency, 1);
        params.set(gaNflushFrequency, 1);
        params.set(gaNselectScores, (int)GAStatistics::AllScores);

        return params;
    }

    static float objectiveFunc(GAGenome &g)
    {
        NetworkGAOptimizer *optimizer = static_cast<NetworkGAOptimizer *>(g.userData());
        Network network = optimizer->blueprint.genome_to_network(g);
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

    static void clean_network(Network &network)
    {
        for (Neuron *n : network.get_neurons())
        {
            delete n;
        }

        for (Synapsis *s : network.get_synapses())
        {
            delete s;
        }
    }

    static bool terminateSimulation(GAGeneticAlgorithm &ga)
    {
        const float convNow = ga.convergence();
        const int maxGens = ga.nGenerations();
        const float convGoal = ga.pConvergence();

        return (ga.generation() >= maxGens || convNow >= convGoal);
    }

    static void printStats(GAGeneticAlgorithm &ga, unsigned int steps)
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
};

#endif // NEURON_GA_OPTIMIZER_H_
