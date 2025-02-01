#ifndef RUN_HPP
#define RUN_HPP

#include "Types.hpp"

double run(const AdjacencyMatrix& adjacencyMatrix, Strategy strategy, const std::vector<std::vector<size_t>>& shuffleSequences, double slope);

#endif // RUN_HPP