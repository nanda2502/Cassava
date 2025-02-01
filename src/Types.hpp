#ifndef TYPES_HPP
#define TYPES_HPP

#include <vector>

using AdjacencyMatrix = std::vector<std::vector<size_t>>;
using State = std::vector<size_t>;
using Parents = std::vector<std::vector<size_t>>;

enum Strategy {
    Random,
    Payoff,
    Proximal,
    Prestige,
    Conformity,
    Perfect
};

struct StateHash {
    std::size_t operator()(const State& state) const {
        std::size_t h = 0;
        for (size_t bit : state) {
            h = h * 31 + bit;
        }
        return h;
    }
};

struct StateEqual {
    bool operator()(const State& lhs, const State& rhs) const {
        return lhs == rhs;
    }
};

struct ParamCombination {
    AdjacencyMatrix adjMatrix;
    Strategy strategy;
    std::vector<std::vector<size_t>> shuffleSequences;
    double slope;
};

#endif // TYPES_HPP