# ifndef MARKOV_HPP
# define MARKOV_HPP

#include "Types.hpp"

class MarkovChain {

public:
MarkovChain
(
    const AdjacencyMatrix& adjMatrix,
    Strategy strategy, 
    std::vector<double> traitFrequencies,
    std::vector<double> stateFrequencies,
    std::vector<double> payoffs, 
    double slope
);

AdjacencyMatrix adjacencyMatrix;
Parents parents;
std::vector<State> allStates;
Strategy strategy;
std::vector<double> traitFrequencies;
std::vector<double> stateFrequencies;
std::vector<double> payoffs;
double slope;
double proportionLearnable;
std::vector<double> computedTraitFrequencies;
std::vector<double> computedStateFrequencies;

std::tuple<std::vector<double>, std::vector<double>, double> run();
static std::vector<State> possibleStates(const AdjacencyMatrix& adjacencyMatrix);
static std::pair<AdjacencyMatrix, std::vector<double>> getSubproblem(
    const AdjacencyMatrix& original, 
    const State& state,
    const std::vector<double>& payoffs);

private:
static std::vector<std::vector<size_t>> parentTraits(const AdjacencyMatrix& adjMatrix);
std::vector<std::vector<double>> buildTransitionMatrix();
std::vector<double> transitionFromState(const State& currentState);
std::vector<bool> learnability(const State& state);
[[nodiscard]] std::vector<std::vector<double>> IMinusQ(const std::vector<std::vector<double>>& reorderedTransitionMatrix) const;
[[nodiscard]] std::vector<std::vector<double>> buildFundamentalMatrix(const std::vector<std::vector<double>>& LU, const std::vector<int>& p) const;
void computeFrequencies(const std::vector<std::vector<double>>& fundamentalMatrix);
double calcPropLearnable(const State& state);
std::vector<double> calculateWeights(const State& state);
std::vector<double> proximalWeights(const State& state);
std::vector<double> prestigeWeights(const State& state);
std::vector<double> conformityWeights(const State& state);
std::vector<double> perfectWeights(const State& repertoire);
};

#endif // MARKOV_HPP