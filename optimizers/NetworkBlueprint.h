#ifndef NETWORK_BLUEPRINT_H_
#define NETWORK_BLUEPRINT_H_

#include <DynamicalSystemLimiter.h>
#include <vector>
#include <ga/GARealGenome.h>

template <typename Network>
class NetworkBlueprint
{

public:
    typedef typename Network::Neuron Neuron;
    typedef typename Network::Synapsis Synapsis;

    typedef DynamicalSystemLimiter<Neuron> NeuronLimiter;
    typedef DynamicalSystemLimiter<Synapsis> SynapsisLimiter;

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

        synapses.push_back(blueprint);
    }

    Network genome_to_network(const GAGenome &genome)
    {
        GARealGenome realGenome = static_cast<const GARealGenome &>(genome);
        Network network;

        for (size_t neuron_idx = 0; neuron_idx < neurons.size(); neuron_idx++)
        {
            Neuron *neuron = genome_to_neuron(realGenome, neuron_idx);
            network.add_neuron(neuron);
        }

        for (size_t synapsis_idx = 0; synapsis_idx < synapses.size(); synapsis_idx++)
        {
            Synapsis *synapsis = genome_to_synapsis(realGenome, network, synapsis_idx);
            network.add_synapsis(synapsis);
        }

        return network;
    }

    GARealAlleleSetArray generate_alleles()
    {
        GARealAlleleSetArray alleleArray;

        for (const NeuronBlueprint &blueprint : neurons)
        {
            add_limits<NeuronLimiter>(blueprint.limits, alleleArray);
        }

        for (const SynapsisBlueprint &blueprint : synapses)
        {
            add_limits<SynapsisLimiter>(blueprint.limits, alleleArray);
        }

        return alleleArray;
    }

private:
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
    std::vector<SynapsisBlueprint> synapses;

    const int neuron_size = Neuron::n_parameters + Neuron::n_variables;
    const int synapsis_size = Synapsis::n_parameters + Synapsis::n_variables;

    Neuron *genome_to_neuron(const GARealGenome &genome, size_t idx)
    {
        typename Neuron::ConstructorArgs args = {};
        const int start = idx * neuron_size;

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

    Synapsis *genome_to_synapsis(const GARealGenome &genome, Network &network, size_t idx)
    {
        const int neurons_end = neurons.size() * neuron_size;
        const int synapsis_start = neurons_end + synapsis_size * idx;

        SynapsisBlueprint blueprint = synapses[idx];
        Neuron *n1 = network.get_neuron(blueprint.n1);
        Neuron *n2 = network.get_neuron(blueprint.n2);

        // this only works for electrical synapsis.
        // There is no common interface for synapsis instanciation
        Synapsis *synapsis = new Synapsis(*n1, Neuron::v, *n2, Neuron::v,
                                          genome.gene(synapsis_start),
                                          genome.gene(synapsis_start + 1));

        return synapsis;
    }

    template <typename Limiter>
    void add_limits(Limiter limiter, GARealAlleleSetArray &alleleArray)
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
};

#endif // NETWORK_BLUEPRINT_H_
