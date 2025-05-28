
#ifndef NEURON_GA_OPTIMIZER_H_
#define NEURON_GA_OPTIMIZER_H_

#define MAX_GENS 1000
#define PRINT_INTERVAL 10
#define INSTANTIATE_REAL_GENOME true

#include <ga/GARealGenome.h>
#include <ga/ga.h>
#include <ga/std_stream.h>
#include <ParamLimiter.h>

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
class NeuronGAOptimizer
{
public:
    typedef TObjective Objective;
    typedef typename Objective::Neuron Neuron;
    typedef ParamLimiter<Neuron> Limiter;

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

    NeuronGAOptimizer(ConstructorArgs &args, Objective &o, Limiter &l)
        : storedObjective(o), ga(createGenome(storedObjective, const_cast<Limiter &>(l)))
    {
        ga.parameters(createParameterList(args.params));
        ga.pReplacement(args.params[parameter::pRepl]);
        ga.pConvergence(args.params[parameter::pConv]);
        unsigned int seed = time(NULL);
        this->ga.initialize(seed);
    }

    // I allow for users to set a seed
    // that ensures the reproducibility of experiments perforned using this feature
    NeuronGAOptimizer(ConstructorArgs &args, Objective &o, Limiter &l, unsigned int seed)
        : storedObjective(o), ga(createGenome(storedObjective, const_cast<Limiter &>(l)))
    {
        ga.parameters(createParameterList(args.params));
        ga.pReplacement(args.params[parameter::pRepl]);
        ga.pConvergence(args.params[parameter::pConv]);
        this->ga.initialize(seed);
    }

    Neuron generate()
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
        return genomeToNeuron(bestGenome);
    }

private:
    Objective storedObjective;
    GASteadyStateGA ga;

    static GARealGenome createGenome(Objective &objective, Limiter &paramLim)
    {
        GARealAlleleSetArray alleles = limitsToAlleleSet(paramLim);
        GARealGenome genome(alleles, objectiveFunc);
        genome.userData(&objective);

        genome.initializer(GARealUniformInitializer);
        genome.crossover(GARealOnePointCrossover);
        genome.mutator(GARealGaussianMutator);

        return genome;
    }

    static GAParameterList createParameterList(float *setupParams)
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

    static GARealAlleleSetArray limitsToAlleleSet(const Limiter &limiter)
    {
        GARealAlleleSetArray alleleArray;

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

        return alleleArray;
    }

    static Neuron genomeToNeuron(const GAGenome &genome)
    {
        typename Neuron::ConstructorArgs args = {};

        GARealGenome realGenome = static_cast<const GARealGenome &>(genome);

        // collect parameters from the genome
        for (int i = 0; i < Neuron::n_parameters; ++i)
        {
            args.params[i] = realGenome.gene(i);
        }

        Neuron neuron(args);

        // collect initial neuron state from the genome
        for (int i = 0; i < Neuron::n_variables; ++i)
        {
            auto var = static_cast<typename Neuron::variable>(i);
            auto value = realGenome.gene(i + Neuron::n_parameters);
            neuron.set(var, value);
        }

        return neuron;
    }

    static float objectiveFunc(GAGenome &g)
    {
        Objective *objective = static_cast<Objective *>(g.userData());
        Neuron n = genomeToNeuron(g);
        float fitness = objective->evaluate(n);

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
