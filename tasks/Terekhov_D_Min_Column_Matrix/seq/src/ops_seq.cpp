#include "Terekhov_D_Min_Column_Matrix/seq/include/ops_seq.hpp"

#include <climits>
#include <cstddef>
#include <vector>

#include "Terekhov_D_Min_Column_Matrix/common/include/common.hpp"

namespace terekhov_d_a_test_task_processes {

TerekhovDTestTaskSEQ::TerekhovDTestTaskSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  if (!in.empty()) {
    GetInput() = in;
  } else {
    GetInput() = InType{};
  }

  GetOutput() = OutType{};
}

bool TerekhovDTestTaskSEQ::ValidationImpl() {
  const auto &input = GetInput();
  if (input.empty()) {
    return false;
  }

  const std::size_t cols = input[0].size();
  if (cols == 0) {
    return false;
  }

  for (const auto &row : input) {
    if (row.size() != cols) {
      return false;
    }
  }

  return true;
}

bool TerekhovDTestTaskSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool TerekhovDTestTaskSEQ::RunImpl() {
  const auto &matrix = GetInput();
  auto &result = GetOutput();

  result.clear();

  if (matrix.empty()) {
    return false;
  }

  const std::size_t rows = matrix.size();
  const std::size_t cols = matrix[0].size();

  result.assign(cols, INT_MAX);

  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      if (matrix[i][j] < result[j]) {
        result[j] = matrix[i][j];
      }
    }
  }

  return true;
}

bool TerekhovDTestTaskSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace terekhov_d_a_test_task_processes
