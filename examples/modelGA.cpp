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
#include <HindmarshRoseModel.h>
#include <RungeKutta4.h>
#include <NeuronOptimizerWrapper.h>
#include <OptimizationObjectiveWrapper.h>
#include <DynamicalSystemLimiter.h>

#include <VoltageDifferenceObjective.h>
#include <NetworkGAOptimizer.h>

#include <vector>
#include <ElectricalSynapsis.h>
#include <NeuronNetwork.h>

typedef RungeKutta4 Integrator;
typedef HindmarshRoseModel<float> HR;
typedef DifferentialNeuronWrapper<HR, Integrator> Neuron;
typedef ElectricalSynapsis<Neuron, Neuron> Synapsis;
typedef NeuronNetwork<Neuron, Synapsis> Network;

typedef VoltageDifferenceObjective<Network> Objective;
typedef NetworkGAOptimizer<Objective> Optimizer;
typedef Optimizer::Blueprint Blueprint;
typedef Blueprint::NeuronLimiter Limiter;

#define STEP 0.001
#define TIME 100
#define INPUT 0.5

double input(double time, const Network &net)
{
  return INPUT;
}

int main(int argc, char **argv)
{

  Neuron::ConstructorArgs neuronArgs;
  neuronArgs.params[Neuron::e] = 3.0;
  neuronArgs.params[Neuron::mu] = 0.002;
  neuronArgs.params[Neuron::S] = 4.0;

  Neuron n(neuronArgs);

  n.set(Neuron::x, -1.3);
  n.set(Neuron::y, -7.0);
  n.set(Neuron::z, -3.0);

  // Initializes the patameters for the optimization objective
  Objective::ConstructorArgs objectiveArgs;
  objectiveArgs.params[Objective::time] = TIME;
  objectiveArgs.params[Objective::step] = STEP;
  objectiveArgs.params[Objective::input] = INPUT;
  Objective objective(objectiveArgs, n, Neuron::x);
  // save the base data to a file
  std::ofstream base_data("model.txt");
  objective.save_voltages(base_data);
  base_data.close();

  // Establishes bounds for the values of each parameter
  Limiter limiter;
  limiter.addLimits(Neuron::e, -10, 10);
  limiter.addLimits(Neuron::mu, 1e-6, 0.1);
  limiter.addLimits(Neuron::S, 0, 10);

  limiter.addLimits(Neuron::x, -10, 10);
  limiter.addLimits(Neuron::y, -20, 10);
  limiter.addLimits(Neuron::z, 0, 10);

  // Initializes the paramters for the optimizer
  Optimizer::ConstructorArgs optimizerArgs;
  optimizerArgs.params[Optimizer::pConv] = 0.99;
  optimizerArgs.params[Optimizer::pRepl] = 0.9;
  optimizerArgs.params[Optimizer::popSize] = 1000;
  optimizerArgs.params[Optimizer::pCross] = 0.9;
  optimizerArgs.params[Optimizer::pMut] = 0.15;
  optimizerArgs.params[Optimizer::nGens] = 500;

  Blueprint b;
  b.add_neuron(limiter);
  b.add_synaptic_input(0, input);

  // Creates the optimizer with the setup
  Optimizer optimizer(optimizerArgs, objective, b, 12345);

  std::cout << "Optimizing neuron:\n";
  Network optimizedNetwork = optimizer.generate();

  Neuron *optimizedNeuron = optimizedNetwork.get_neuron(0);

  std::cout << "Optimized Neuron Parameters:\n";
  std::cout << "e   = " << optimizedNeuron->get(Neuron::e) << "\n";
  std::cout << "mu  = " << optimizedNeuron->get(Neuron::mu) << "\n";
  std::cout << "S   = " << optimizedNeuron->get(Neuron::S) << "\n";

  std::cout << "\nNeuron State Variables:\n";
  std::cout << "x = " << optimizedNeuron->get(Neuron::x) << "\n";
  std::cout << "y = " << optimizedNeuron->get(Neuron::y) << "\n";
  std::cout << "z = " << optimizedNeuron->get(Neuron::z) << "\n";

  std::ofstream data("model_recreated.txt");
  optimizedNetwork.simulate(TIME, STEP, data, Neuron::x, Synapsis::i1);
  data.close();

  return 0;
}
