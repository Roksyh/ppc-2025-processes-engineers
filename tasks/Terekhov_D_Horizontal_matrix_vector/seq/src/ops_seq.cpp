#include "terekhov_d_horizontal_matrix_vector/seq/include/ops_seq.hpp"

#include <cstddef>
#include <utility>
#include <vector>

namespace terekhov_d_horizontal_matrix_vector {

TerekhovDHorizontalMatrixVectorSEQ::TerekhovDHorizontalMatrixVectorSEQ(InType in) : input_(std::move(in)) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetOutput() = std::vector<double>();
}

bool TerekhovDHorizontalMatrixVectorSEQ::ValidationImpl() {
  const auto &matrix_a = input_.first;
  const auto &vector_b = input_.second;

  if (matrix_a.empty() || vector_b.empty()) {
    return false;
  }

  size_t cols_a = matrix_a[0].size();
  for (size_t i = 1; i < matrix_a.size(); i++) {
    if (matrix_a[i].size() != cols_a) {
      return false;
    }
  }

  return cols_a == vector_b.size();
}

bool TerekhovDHorizontalMatrixVectorSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool TerekhovDHorizontalMatrixVectorSEQ::RunImpl() {
  const auto &matrix_a = input_.first;
  const auto &vector_b = input_.second;

  size_t rows_a = matrix_a.size();
  size_t cols_a = matrix_a[0].size();

  auto &output = GetOutput();
  output = std::vector<double>(rows_a, 0.0);

  for (size_t i = 0; i < rows_a; i++) {
    for (size_t j = 0; j < cols_a; j++) {
      output[i] += matrix_a[i][j] * vector_b[j];
    }
  }

  return true;
}

bool TerekhovDHorizontalMatrixVectorSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace terekhov_d_horizontal_matrix_vector
