#include "Payoffs.hpp"
#include "Types.hpp"
#include "Markov.hpp"
#include "Run.hpp"


double run(const AdjacencyMatrix& adjacencyMatrix, Strategy strategy, const std::vector<std::vector<size_t>>& shuffleSequences, double slope) {
    auto states = MarkovChain::possibleStates(adjacencyMatrix);
    std::vector<double> initialStateFrequencies(states.size(), 1.0);
    std::vector<double> initialTraitFrequencies(adjacencyMatrix.size(), 1.0);

    double weightedSum = 0.0;

    for (const auto& shuffleSequence : shuffleSequences) {
        auto payoffs = generatePayoffs(shuffleSequence);

        // Do first run with random strategy to get background frequencies
        auto [baseTraitFrequencies, baseStateFrequencies, __] = MarkovChain(
            adjacencyMatrix, 
            Strategy::Random, 
            initialTraitFrequencies,
            initialStateFrequencies,  
            payoffs, 
            slope
            ).run();

        // Run the Markov chain with the given strategy
        auto [traitFrequencies, stateFrequencies, ___] = MarkovChain(
            adjacencyMatrix, 
            strategy, 
            baseTraitFrequencies,
            baseStateFrequencies,  
            payoffs, 
            slope
            ).run();
        

        // Calculate the average proportion of learnable traits over time
        for (size_t stateIdx = 0; stateIdx < states.size(); ++stateIdx) {
            const auto& state = states[stateIdx];
            double stateFreq = stateFrequencies[stateIdx];
            
            // Count number of unlearned traits
            size_t unlearned = 0;
            for (size_t i : state) {
                if (i == 0) unlearned++;
            }

            // Calculate proportion learnable for this state
            double propLearnable;
            if (unlearned > 1) {
                auto [subproblem, subpayoffs] = MarkovChain::getSubproblem(adjacencyMatrix, state, payoffs);
                auto [___, ____, pl] = MarkovChain(
                    subproblem, 
                    strategy,
                    baseTraitFrequencies,
                    baseStateFrequencies,
                    subpayoffs,
                    slope 
                    ).run();
                propLearnable = pl;
            } else if (unlearned == 1) {
                propLearnable = 1.0;  // If only one trait is unlearned, it must be learnable
            } else {
                propLearnable = 0.0;  // If no traits are unlearned, nothing is learnable
            }
            
            // Add this state's contribution to the total, weighted by the time spent in that state and averaged over all payoff shuffles
            weightedSum += (stateFreq * propLearnable)/shuffleSequences.size();
        }
    }
    
    return weightedSum;
}