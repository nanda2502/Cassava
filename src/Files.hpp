#ifndef FILES_HPP
#define FILES_HPP

#include "Types.hpp"
#include <string>

std::vector<AdjacencyMatrix> readAdjacencyMatrices(int numMatrices);
std::string formatResults(
    const AdjacencyMatrix& adjacencyMatrix,
    Strategy strategy,
    double mean,
    double slope
);
std::string adjMatrixToBinaryString(const AdjacencyMatrix& adjMatrix);
AdjacencyMatrix binaryStringToAdjacencyMatrix(const std::string& str);
#endif // FILES_HPP