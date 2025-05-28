#ifndef AMPLITUDE_OBJECTIVE_H_
#define AMPLITUDE_OBJECTIVE_H_

#include "DifferentialNeuronWrapper.h"

template <typename TNeuron>
class AmplitudeObjective
{
public:
    typedef typename Neuron::precission_t precission_t;

    precission_t evaluate(Neuron n)
    {
    }
};

#endif // AMPLITUDE_OBJECTIVE_H_
