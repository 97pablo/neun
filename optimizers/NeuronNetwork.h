#ifndef NEURON_NETWORK_H_
#define NEURON_NETWORK_H_

#include <vector>
#include <functional>
#include <memory>

template <typename TNeuron, typename TSynapsis>
class NeuronNetwork
{
public:
    typedef TNeuron Neuron;
    typedef TSynapsis Synapsis;

private:
    std::vector<Neuron *> neurons;
    std::vector<Synapsis *> synapsises;

    // External input is stored as functions, that allows behabiors like
    // a ramping voltage, or a fixed voltage from a specific point in time
    // The network is also included as an input
    // that allows for functions that depend on the state of any neuron.
    std::vector<std::vector<std::function<double(double, const NeuronNetwork<Neuron, Synapsis> &)>>> inputs;

public:
    /**
     * @brief adds a neuron to the network
     *
     * @param n the neuron to be added
     * @return int an identificator, can be used to work with neurons within the network
     */
    int add_neuron(Neuron *n)
    {
        neurons.push_back(n);
        inputs.emplace_back();
        return neurons.size() - 1;
    }

    /**
     * @brief adds a synapsis to the network
     *
     * @param s the synapsis to be added
     * @return int an identificator, can be used to work with neurons within the network
     */
    int add_synapsis(Synapsis *s)
    {
        synapsises.push_back(s);
        return neurons.size() - 1;
    }

    /**
     * @brief adds a variable input for a neuron in the network
     *
     * @param neuron_id identifier of the neuron that the input is added to
     * @param input
     */
    int add_synaptic_input(int neuron_id, std::function<double(double, const NeuronNetwork<Neuron, Synapsis> &)> input)
    {
        inputs[neuron_id].push_back(input);
        return inputs.size() - 1;
    }

    std::vector<Neuron *> get_neurons()
    {
        return neurons;
    }

    std::vector<Synapsis *> get_synapsises()
    {
        return synapsises;
    }

    std::vector<std::function<double(double, const NeuronNetwork<Neuron, Synapsis> &)>> get_inputs(int neuron_id)
    {
        return inputs[neuron_id];
    }

    Neuron *get_neuron(int neuron_id)
    {
        return neurons[neuron_id];
    }

    Synapsis *get_synapsis(int synapsis_id)
    {
        return synapsises[synapsis_id];
    }

    /**
     * @brief advances the state of the simulation by the time speficied
     *
     * @param step the time by witch the simulation advances
     */
    void step(double step)
    {
        // simulate synapsis
        for (Synapsis *s : synapsises)
        {
            s->step(step);
        }

        for (int i = 0; i < (int)neurons.size(); i++)
        {
            Neuron *n = neurons[i];

            // simulate inputs
            for (auto input_func : inputs[i])
            {
                double input_value = input_func(step, *this);
                n->add_synaptic_input(input_value);
            }

            // simulate neuron
            n->step(step);
        }
    }

    /**
     * @brief advances the simulation by the specified time,
     * logging the state in step intervals to an output stream
     *
     * @param time the time the simulation lasts
     * @param step size of the simulation steps
     * @param out the ouput stream results are written to
     */
    void simulate(double simulation_time, double step, std::ostream &out)
    {
        for (double time = 0; time < simulation_time; time += step)
        {
            this->step(step);

            // print neuron and synapsis state
            out << time << " ";
            for (const Neuron *n : neurons)
            {
                out << n->get(Neuron::v) << " ";
            }
            for (const Synapsis *s : synapsises)
            {
                out << s->get(Synapsis::i1) << " ";
            }
            out << std::endl;
        }
    }
};

#endif // NEURON_NETWORK_H_