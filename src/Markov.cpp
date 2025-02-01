#include "Markov.hpp"
#include "LinAlg.hpp"


#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>
#include <tuple>

MarkovChain::MarkovChain
(
    const AdjacencyMatrix& adjMatrix,
    Strategy strategy, 
    std::vector<double> traitFrequencies,
    std::vector<double> stateFrequencies,
    std::vector<double> payoffs, 
    double slope
) {
    this->strategy = strategy;
    this->payoffs = std::move(payoffs);
    this->slope = slope;
    this->traitFrequencies = std::move(traitFrequencies);
    this->stateFrequencies = std::move(stateFrequencies);
    adjacencyMatrix = adjMatrix;
    parents = parentTraits(adjMatrix);
    allStates = possibleStates(adjMatrix);
    computedTraitFrequencies = std::vector<double>(adjMatrix.size());
    computedStateFrequencies = std::vector<double>(allStates.size());
    State initialState(adjMatrix.size(), 0);
    initialState[0] = 1;
    proportionLearnable = calcPropLearnable(initialState);
}

double MarkovChain::calcPropLearnable(const State& state) {
    // Get vector of learnable traits
    std::vector<bool> canLearn = learnability(state);
    
    // Count total unlearned traits
    size_t totalUnlearned = 0;
    size_t learnableCount = 0;
    
    for (size_t i = 0; i < state.size(); ++i) {
        if (state[i] == 0) { // If trait is not learned
            totalUnlearned++;
            if (canLearn[i]) {
                learnableCount++;
            }
        }
    }
    
    // Return 0 if there are no unlearned traits to avoid division by zero
    return totalUnlearned == 0 ? 0.0 : static_cast<double>(learnableCount) / totalUnlearned;
}

std::pair<AdjacencyMatrix, std::vector<double>> MarkovChain::getSubproblem(
    const AdjacencyMatrix& original, 
    const State& state,
    const std::vector<double>& payoffs) {
    
    size_t n = state.size();
    
    // Count unlearned nodes and create mapping
    std::vector<int> newIndices(n, -1);
    int newIndex = 1; // Start at 1 since 0 is root
    
    for (size_t i = 1; i < n; i++) {
        if (state[i] == 0) {
            newIndices[i] = newIndex++;
        }
    }
    
    // Create new matrix (size is number of unlearned nodes + root)
    AdjacencyMatrix submat(newIndex, std::vector<size_t>(newIndex, 0));
    std::vector<double> subpayoffs(newIndex);
    
    // Initialize root node payoff with sum of all learned traits' payoffs
    subpayoffs[0] = 0.0;
    for (size_t i = 0; i < n; i++) {
        if (state[i] == 1) {
            subpayoffs[0] += payoffs[i];
        }
    }
    
    // For each unlearned node
    for (size_t i = 1; i < n; i++) {
        if (state[i] == 0) { // If this is an unlearned node
            // Set payoff for this node in subproblem
            subpayoffs[newIndices[i]] = payoffs[i];
            
            // Check if it has any learned prerequisites
            bool hasLearnedPrereq = false;
            for (size_t j = 0; j < n; j++) {
                if (state[j] == 1 && original[j][i] == 1) {
                    hasLearnedPrereq = true;
                    break;
                }
            }
            
            // If it has learned prerequisites, it connects to root
            if (hasLearnedPrereq) {
                submat[0][newIndices[i]] = 1;
            }
            
            // Check connections to other unlearned nodes
            for (size_t j = 1; j < n; j++) {
                if (state[j] == 0 && original[i][j] == 1) {
                    submat[newIndices[i]][newIndices[j]] = 1;
                }
            }
        }
    }
    
    return {submat, subpayoffs};
}

std::vector<std::vector<size_t>> MarkovChain::parentTraits(const AdjacencyMatrix& adjMatrix) {
    std::vector<std::vector<size_t>> allParents;

    for (size_t trait = 0; trait < adjMatrix.size(); ++trait) {
        
        std::vector<size_t> parents;
        size_t n = adjMatrix.size();

        for (size_t parent = 0; parent < n; ++parent) {
            if (adjMatrix[parent][trait] == 1) {
                parents.push_back(parent);
            }
        }
        allParents.push_back(parents);
    }
    return allParents;
}


std::vector<State> MarkovChain::possibleStates(const AdjacencyMatrix& adjacencyMatrix) {
    size_t n = adjacencyMatrix.size();
    
    // Start with just the root trait mastered
    State initialState(n, 0);
    initialState[0] = 1;
    
    std::queue<State> queue;
    std::unordered_set<State, StateHash, StateEqual> visited;
    std::vector<State> states;
    
    queue.push(initialState);
    
    while (!queue.empty()) {
        State currentState = queue.front();
        queue.pop();
        
        // If we haven't seen this state yet
        if (visited.find(currentState) == visited.end()) {
            visited.insert(currentState);
            states.push_back(currentState);
            
            // Try learning each unlearned trait
            for (size_t trait = 0; trait < n; ++trait) {
                if (currentState[trait] == 0) { // If trait is not learned
                    bool canLearn = true;
                    
                    // Check if all prerequisites are learned
                    for (size_t prereq = 0; prereq < n; ++prereq) {
                        if (adjacencyMatrix[prereq][trait] == 1 && currentState[prereq] == 0) {
                            canLearn = false;
                            break;
                        }
                    }
                    
                    // If we can learn this trait, create a new state
                    if (canLearn) {
                        State nextState = currentState;
                        nextState[trait] = 1;
                        queue.push(nextState);
                    }
                }
            }
        }
    }
    
    return states;
}

std::vector<double> toProbabilityDistribution(std::vector<double>& input) {
    if (input.empty()) return input;
    
    double sum = std::accumulate(input.begin(), input.end(), 0.0);
    std::vector<double> result(input.size());
    std::transform(input.begin(), input.end(), result.begin(),
        [sum](double x) { return sum != 0.0 ? x / sum : 0.0; });
    
    return result;
}

std::vector<bool> MarkovChain::learnability(const State& state) {
    std::vector<bool> canLearn(state.size(), false);
    
    for (size_t trait = 0; trait < state.size(); ++trait) {
        // If trait is already known, we can't learn it
        if (state[trait] == 1) continue;
        
        // Check if all parents are known
        bool allParentsKnown = true;
        for (size_t parent : parents[trait]) {
            if (state[parent] == 0) {
                allParentsKnown = false;
                break;
            }
        }
        
        canLearn[trait] = allParentsKnown;
    }
    
    return canLearn;
}

double computeDelta(const State& r, const State& s) {
    // count the number of traits that are present in target state s but not in current state r
    return std::inner_product(s.begin(), s.end(), r.begin(), 0.0,
        std::plus<>(), [](size_t s_i, size_t r_i) { return (s_i == 1 && r_i == 0) ? 1 : 0; });
}

std::vector<double> MarkovChain::proximalWeights(const State& state) {
    std::vector<double> weights(state.size(), 0.0);

    for (size_t trait = 0; trait < state.size(); ++trait) {
        if (state[trait] == 0) {  // Only consider unlearned traits
            for (size_t stateidx = 0; stateidx < allStates.size(); ++stateidx) {
                const auto& demoState = allStates[stateidx];
                if (demoState[trait] == 1) {  // If the state has trait j learned
                    double delta = computeDelta(state, demoState);
                    if (delta > 0) {
                        weights[trait] += stateFrequencies[stateidx] * std::pow(slope, -delta); 
                    }
                }
            }
        }
    }
    return weights;
}

std::vector<double> MarkovChain::prestigeWeights(const State& state) {
    std::vector<double> weights(state.size(), 0.0);

    for (size_t trait = 0; trait < state.size(); ++trait) {
        if (state[trait] == 0) {  // Only consider unlearned traits
            for (size_t stateidx = 0; stateidx < allStates.size(); ++stateidx) {
                const auto& demoState = allStates[stateidx];
                if (demoState[trait] == 1) {  // If the state has trait j learned
                    double delta = computeDelta(state, demoState);
                    if (delta > 0) {
                        weights[trait] += stateFrequencies[stateidx] * std::pow(slope, delta); 
                    }
                }
            }
        }
    }
    return weights;
}

double S_curve(double x, double total, double slope) {
    // between 0 and 1;
    return 1/(1+std::exp(-slope * ((x/total)-0.5)));
}

std::vector<double> MarkovChain::conformityWeights(
    const State& state
) {
    double total = 0.0;

    //Only count unlearned traits, so that the normalization works
    for (size_t i = 0; i < traitFrequencies.size(); i++) {
        total += state[i] == 1 ? 0.0 : traitFrequencies[i];
    }

    std::vector<double> w_star(traitFrequencies.size());
    
    std::transform(traitFrequencies.begin(), traitFrequencies.end(), w_star.begin(), [total, this](double f) {return S_curve(f, total, slope);});

    return w_star;
}

std::vector<double> MarkovChain::perfectWeights(
    const State& repertoire
) {
    // always pick the learnable trait with the highest payoff
    auto learnable = learnability(repertoire);
    double highestPayoff = 0.0;
    double bestTraitidx = 0;
    std::vector<double> weights(repertoire.size(), 0.0);
    for (size_t trait = 0; trait < repertoire.size(); ++trait) {
        if (learnable[trait] && payoffs[trait] > highestPayoff) {
            highestPayoff = payoffs[trait];
            bestTraitidx = trait;
        }

    }
    weights[bestTraitidx] = 1.0;
    return weights;
}


std::vector<double> MarkovChain::calculateWeights(const State& state) {
    switch (strategy) {
        case Random:
            return traitFrequencies;
        case Payoff:
        {
            std::vector<double> result(payoffs.size());
            std::transform(payoffs.begin(), payoffs.end(), traitFrequencies.begin(), result.begin(),
               [this](double payoff, double traitFrequency) {
                   return traitFrequency * std::pow(payoff, slope);
               });
            return result;
        }
        case Proximal:
            return proximalWeights(state);
        case Prestige:
            return prestigeWeights(state);
        case Conformity:
            return conformityWeights(state);
        case Perfect:
            return perfectWeights(state);
        default:
            throw std::runtime_error("Invalid strategy");
    }
}

std::vector<double> MarkovChain::transitionFromState(const State& currentState) {
    std::vector<double> probabilities(allStates.size(), 0.0);
    std::vector<double> weights(currentState.size(), 1.0);
    
    auto canLearn = learnability(currentState);
    
    // Zero out weights for learned traits
    for (size_t trait = 0; trait < currentState.size(); ++trait) {
        if (currentState[trait] == 1) weights[trait] = 0.0;
    }        

    weights = toProbabilityDistribution(weights);

    // Find current state index
    auto currentStateIt = std::find(allStates.begin(), allStates.end(), currentState);
    size_t currentStateIdx = currentStateIt - allStates.begin();
    
    // Start with probability of staying in current state
    double stayProb = 0.0;

    // Calculate transitions for each trait
    for (size_t trait = 0; trait < currentState.size(); ++trait) {
        if (canLearn[trait] && weights[trait] > 0.0) {
            // Create new state and find its index
            auto newState = currentState;
            newState[trait] = 1;
            auto newStateIt = std::find(allStates.begin(), allStates.end(), newState);
            size_t newStateIdx = newStateIt - allStates.begin();
            
            probabilities[newStateIdx] = weights[trait];
        } else {
            stayProb += weights[trait];
        }
    }
    
    probabilities[currentStateIdx] = stayProb;
    return probabilities;
}


std::vector<std::vector<double>> MarkovChain::buildTransitionMatrix() {
    std::vector<std::vector<double>> transitionMatrix;
    transitionMatrix.reserve(allStates.size());

    for (const auto& state : allStates) {
        transitionMatrix.push_back(transitionFromState(state));
    }
    
    return transitionMatrix;
}

std::vector<std::vector<double>> MarkovChain::IMinusQ(const std::vector<std::vector<double>>& transitionMatrix) const  {
    // The BFS that generates the states guarantees that the absorbing state will be the last state in the matrix, so everything before it is transient.
    int numTransientStates = allStates.size() - 1;

    // Extract the Q matrix from the reordered transition matrix. The Q matrix represents the transition probabilities between transient states.
    std::vector<std::vector<double>> qMatrix(numTransientStates, std::vector<double>(numTransientStates));
    for (int i = 0; i < numTransientStates; ++i) {
        for (int j = 0; j < numTransientStates; ++j) {
            qMatrix[i][j] = transitionMatrix[i][j];
        }
    }

    // Subtract the Q matrix from the identity matrix to get I - Q.
    std::vector<std::vector<double>> iMinusQ(numTransientStates, std::vector<double>(numTransientStates));
    for (int i = 0; i < numTransientStates; ++i) {
        for (int j = 0; j < numTransientStates; ++j) {
            iMinusQ[i][j] = (i == j ? 1.0 : 0.0) - qMatrix[i][j];
        }
    }

    return iMinusQ;
}

std::vector<std::vector<double>> MarkovChain::buildFundamentalMatrix(const std::vector<std::vector<double>>& LU, const std::vector<int>& p) const{
    int n = allStates.size() - 1;
    std::vector<std::vector<double>> fundamentalMatrix(n, std::vector<double>(n));

    for (int i = 0; i < n; ++i) {
        std::vector<double> b(n, 0.0);
        b[i] = 1.0;
        std::vector<double> column = solveUsingLU(LU, p, b);

        for (int j = 0; j < n; ++j) {
            fundamentalMatrix[j][i] = column[j];
        }
    }
    return fundamentalMatrix;
}

void MarkovChain::computeFrequencies(const std::vector<std::vector<double>>& fundamentalMatrix) {
    double totalTransientTime = std::accumulate(fundamentalMatrix[0].begin(), fundamentalMatrix[0].end(), 0.0);

    for (size_t trait = 1; trait < adjacencyMatrix.size(); ++trait) {
        double timeTraitKnown = 0.0;
        for (size_t state = 0; state < allStates.size(); ++state) {
            timeTraitKnown += fundamentalMatrix[0][state];
        }
        computedTraitFrequencies[trait] = timeTraitKnown / totalTransientTime;    
    }

    for (size_t j = 1; j < adjacencyMatrix.size(); ++j) {
        if (computedTraitFrequencies[j] == 0.0) {
            computedTraitFrequencies[j] = 1e-5;
        }
    }

    toProbabilityDistribution(computedTraitFrequencies);

    for (size_t state = 0; state < allStates.size() - 1; ++state) {
        computedStateFrequencies[state] = fundamentalMatrix[0][state] / totalTransientTime;
    }

    computedStateFrequencies[allStates.size() - 1] = 1e-5;
    toProbabilityDistribution(computedStateFrequencies);
}

std::tuple<std::vector<double>, std::vector<double>, double> MarkovChain::run() {
    std::vector<std::vector<double>> transitionMatrix = buildTransitionMatrix();
    auto iMinusQMatrix = IMinusQ(transitionMatrix);
    auto [LU, p] = decomposeLU(iMinusQMatrix);
    auto fundamentalMatrix = buildFundamentalMatrix(LU, p);
    computeFrequencies(fundamentalMatrix);
    return {computedTraitFrequencies, computedStateFrequencies, proportionLearnable};
}
