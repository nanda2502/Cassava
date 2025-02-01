#include "Files.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <sstream>


AdjacencyMatrix binaryStringToAdjacencyMatrix(const std::string& str) {
    std::string binaryStr = str;

    int n = static_cast<int>(std::sqrt(binaryStr.size()));

    if (n == 0) throw std::invalid_argument("Invalid adjmat string: " + str);

    AdjacencyMatrix matrix(n, std::vector<size_t>(n));
    for (int row = 0; row < n; ++row) {
        for (int column = 0; column < n; ++column) {
            // Convert ASCII '0'/'1' to actual 0/1 values
            matrix[row][column] = binaryStr[row * n + column] == '1' ? 1 : 0;
        }
    }

    return matrix;
}

std::string adjMatrixToBinaryString(const AdjacencyMatrix& adjMatrix) {
    std::string binaryString;
    binaryString.reserve(adjMatrix.size() * adjMatrix[0].size());

    for (const auto& row : adjMatrix) {
        for (size_t entry : row) {
            binaryString += entry == 1 ? '1' : '0';
        }
    }
    return binaryString;
}

std::string strategyToString(Strategy strategy) {
    switch (strategy) {
        case Random:
            return "Random";
        case Payoff:
            return "Payoff";
        case Proximal:
            return "Proximal";
        case Prestige:
            return "Prestige";
        case Conformity:
            return "Conformity";
        case Perfect:
            return "Perfect";
        default:
            throw std::invalid_argument("Unknown strategy");
    }
}

std::string formatResults(
    const AdjacencyMatrix& adjacencyMatrix,
    Strategy strategy,
    double mean,
    double slope
) {
    std::ostringstream oss;
    oss << adjMatrixToBinaryString(adjacencyMatrix) << ','  <<
    strategyToString(strategy) << ',' <<
    mean << ',' << 
    slope << '\n';
    return oss.str();
}

std::vector<AdjacencyMatrix> readAdjacencyMatrices(int n) {
    std::string filePath = "../adj_mat_" + std::to_string(n) + ".csv";
    std::ifstream file(filePath);
    if (!file.is_open()) throw std::runtime_error("Could not open file " + filePath);

    std::vector<AdjacencyMatrix> matrices;
    std::string line;
    while (std::getline(file, line)) {
        matrices.push_back(binaryStringToAdjacencyMatrix(line));
    }
    
    std::cout << "Loaded " << matrices.size() << " adjacency matrices." << '\n';

    return matrices;
}