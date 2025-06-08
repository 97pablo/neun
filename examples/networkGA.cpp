/*************************************************************

Copyright (c) 2006, Fernando Herrero Carrón
Copyright (c) 2020, Angel Lareo <angel.lareo@gmail.com>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.
    * Neither the name of the author nor the names of his contributors
      may be used to endorse or promote products derived from this
      software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*************************************************************/

#include <DifferentialNeuronWrapper.h>
#include <HodgkinHuxleyModel.h>
#include <RungeKutta4.h>
#include <DynamicalSystemLimiter.h>

#include <AmplitudeObjective.h>
#include <ElectricalSynapsis.h>
#include <NeuronNetwork.h>
#include <NetworkGAOptimizer.h>

#include <vector>

typedef RungeKutta4 Integrator;
typedef DifferentialNeuronWrapper<HodgkinHuxleyModel<float>, Integrator> Neuron;
typedef ElectricalSynapsis<Neuron, Neuron> Synapsis;
typedef NeuronNetwork<Neuron, Synapsis> Network;

typedef DynamicalSystemLimiter<Neuron> NeuronLimiter;
typedef DynamicalSystemLimiter<Synapsis> SynapsisLimiter;
typedef AmplitudeObjective<Network> Objective;
typedef NetworkGAOptimizer<Objective> Optimizer;

int main(int argc, char **argv)
{
  // Initializes the patameters for the optimization objective
  Objective::ConstructorArgs objectiveArgs;
  objectiveArgs.params[Objective::time] = 100;
  objectiveArgs.params[Objective::step] = 0.001;
  objectiveArgs.params[Objective::peak_tolerance] = 0.3;
  objectiveArgs.params[Objective::amplitude] = 10;
  objectiveArgs.params[Objective::amp_tolerance] = 0.15;
  objectiveArgs.params[Objective::n_peaks] = 5;
  Objective objective(objectiveArgs);

  // Establishes bounds for the values of each parameter
  NeuronLimiter neuron_limiter;
  neuron_limiter.addLimits(Neuron::cm, 0, 2.0);
  neuron_limiter.addLimits(Neuron::vna, 30.0, 70.0);
  neuron_limiter.addLimits(Neuron::vk, -100.0, -60.0);
  neuron_limiter.addLimits(Neuron::vl, -80.0, -40.0);
  neuron_limiter.addLimits(Neuron::gna, 0, 100.0);
  neuron_limiter.addLimits(Neuron::gk, 0, 100.0);
  neuron_limiter.addLimits(Neuron::gl, 0, 1.0);

  neuron_limiter.addLimits(Neuron::v, -100.0, 5);
  neuron_limiter.addLimits(Neuron::m, 0.0, 0.2);
  neuron_limiter.addLimits(Neuron::n, 0.4, 0.8);
  neuron_limiter.addLimits(Neuron::h, 0.0, 1.0);

  SynapsisLimiter syn_limiter;
  syn_limiter.addLimits(Synapsis::g1, -0.003, -0.001);
  syn_limiter.addLimits(Synapsis::g2, -0.003, -0.001);
  syn_limiter.addLimits(Synapsis::i1, 0, 0);
  syn_limiter.addLimits(Synapsis::i1, 0, 0);

  // Initializes the paramters for the optimizer
  Optimizer::ConstructorArgs optimizerArgs;
  optimizerArgs.params[Optimizer::pConv] = 0.99;
  optimizerArgs.params[Optimizer::pRepl] = 0.6;
  optimizerArgs.params[Optimizer::popSize] = 500;
  optimizerArgs.params[Optimizer::pCross] = 0.9;
  optimizerArgs.params[Optimizer::pMut] = 0.15;
  optimizerArgs.params[Optimizer::nGens] = 500;
  optimizerArgs.params[Optimizer::nElite] = 1;

  // Creates the optimizer with the setup
  Optimizer optimizer(optimizerArgs, objective, 12345);
  auto n1 = optimizer.add_neuron(neuron_limiter);
  auto n2 = optimizer.add_neuron(neuron_limiter);
  optimizer.add_synapsis(n1, n2, syn_limiter);

  std::cout << "Optimizing neuron:\n";
  Network optimizedNetwork = optimizer.generate();

  std::ofstream data("example_network2.txt");
  optimizedNetwork.simulate(100, 0.001, data);

  for (Neuron *n : optimizedNetwork.get_neurons())
  {
    delete n;
  }

  for (Synapsis *s : optimizedNetwork.get_synapsises())
  {
    delete s;
  }

  return 0;
}
