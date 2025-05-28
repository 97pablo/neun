
#ifndef OBJECTIVEWRAPPER_H_
#define OBJECTIVEWRAPPER_H_

template <typename Wrapee>
class OptimizationObjectiveWrapper : public Wrapee
{
public:
    typedef typename Wrapee::Neuron Neuron;
    typedef typename Wrapee::ConstructorArgs ConstructorArgs;

    OptimizationObjectiveWrapper(ConstructorArgs &args)
        : Wrapee(args) {}

    double evaluate(Neuron n)
    {
        return Wrapee::evaluate(n);
    }
};

#endif // OBJECTIVEWRAPPER_H_
