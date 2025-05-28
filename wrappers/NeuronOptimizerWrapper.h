
#ifndef NEURON_OPTIMIZER_WRAPPER_H_
#define NEURON_OPTIMIZER_WRAPPER_H_

template <typename Wrapee>
class NeuronOptimizerWrapper : public Wrapee
{
public:
    typedef typename Wrapee::Neuron Neuron;
    typedef typename Wrapee::Objective Objective;
    typedef typename Wrapee::ConstructorArgs ConstructorArgs;
    typedef typename Wrapee::Limiter Limiter;

    NeuronOptimizerWrapper(ConstructorArgs &args, Objective o, Limiter l)
        : Wrapee(args, o, l) {}

    NeuronOptimizerWrapper(ConstructorArgs &args, Objective o, Limiter l, unsigned int seed)
        : Wrapee(args, o, l, seed) {}

    Neuron generate() { return Wrapee::generate(); }
};

#endif // NEURON_OPTIMIZER_WRAPPER_H_
