#include "Run.hpp"
#include "Files.hpp"
#include "Types.hpp"
#include <cstddef>
#include <fstream>
#include <iostream>
#include <numeric>

size_t factorial(size_t num) {
    size_t result = 1;
    for (size_t i = 2; i <= num; ++i) {
        result *= i;
    }
    return result;
}

std::vector<double> returnSlopeVector(Strategy strategy) {
    switch (strategy) {
        case Random:
            return {0.0};
        case Perfect:
            return {0.0};
        default:
            return {0.0, 1.0, 1.25, 2.0 , 2.5, 5.0, 10.0, 20.0, 40.0};	

    }
}

std::vector<ParamCombination> makeCombinations(
    const std::vector<AdjacencyMatrix>& adjacencyMatrices,
    const std::vector<Strategy>& strategies,
    const std::vector<std::vector<size_t>>& shuffleSequences,
    const std::vector<std::vector<size_t>>& noshuffleVec
) {
    std::vector<ParamCombination> combinations;
    for (const auto& adjMat : adjacencyMatrices) {
        for (const auto& strategy : strategies) {
            auto slopes = returnSlopeVector(strategy);
            if (strategy == Payoff) {
                for (const auto& slope : slopes) {
                    combinations.push_back({adjMat, strategy, shuffleSequences, slope});
                }
            } else {
                for (const auto& slope : slopes) {
                    combinations.push_back({adjMat, strategy, noshuffleVec, slope});
                }
            }
        }
    } 
    return combinations;
}

int main() {
    auto adjacencyMatrices = readAdjacencyMatrices(8);
    std::vector<double> means;

    std::vector<AdjacencyMatrix> subProblems;
    std::vector<double> subProblemProportions;

    std::vector<size_t> noshuffle(adjacencyMatrices[0].size() -1);
    std::iota(noshuffle.begin(), noshuffle.end(), 0);
    std::vector<std::vector<size_t>> noshuffleVec = {noshuffle};

    std::vector<size_t> perm(adjacencyMatrices[0].size() - 1);
    size_t sequenceCount = factorial(perm.size());
    std::vector<std::vector<size_t>> shuffleSequences;
    shuffleSequences.reserve(sequenceCount);  

    do {
        shuffleSequences.push_back(perm);
    } while (std::next_permutation(perm.begin(), perm.end()));

    std::vector<Strategy> strategies = {
            Strategy::Random,
            Strategy::Payoff,
            Strategy::Proximal,
            Strategy::Prestige,
            Strategy::Conformity,
            Strategy::Perfect
        };

    auto combinations = makeCombinations(adjacencyMatrices, strategies, shuffleSequences, noshuffleVec);
    std::vector<double> result(combinations.size());

    std::vector<size_t> indices(combinations.size());
    std::iota(indices.begin(), indices.end(), 0);

    #pragma omp parallel for
    for (size_t i = 0; i < combinations.size(); ++i) {
        result[i] = run(combinations[i].adjMatrix, combinations[i].strategy, combinations[i].shuffleSequences, combinations[i].slope);
    }

    std::string csvHeader = "adj_mat, strategy, prop_learnable, slope\n";
    std::vector<std::string> csvData;
    csvData.push_back(csvHeader);

    for (size_t i = 0; i < combinations.size(); ++i) {
        auto formattedResult = formatResults(combinations[i].adjMatrix, combinations[i].strategy, result[i], combinations[i].slope);
        csvData.push_back(formattedResult);
    }

    std::ofstream file("../results.csv");
    for (const auto& line : csvData) {
        file << line;
    }

    std::cout << "Results written to results.csv" << '\n';
}