#include "Terekhov_D_Min_Column_Matrix/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <ranges>
#include <vector>

#include "Terekhov_D_Min_Column_Matrix/common/include/common.hpp"

namespace terekhov_d_a_test_task_processes {

TerekhovDTestTaskSEQ::TerekhovDTestTaskSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType temp_in = in;
  GetInput().swap(temp_in);
  OutType temp_out;
  GetOutput().swap(temp_out);
}

bool TerekhovDTestTaskSEQ::ValidationImpl() {
  const InType &matrix = GetInput();
  if (matrix.empty()) {
    return false;
  }

  const std::size_t cols = matrix[0].size();
  if (cols == 0) {
    return false;
  }

  return std::ranges::all_of(matrix, [cols](const auto &row) { return row.size() == cols; });
}

bool TerekhovDTestTaskSEQ::PreProcessingImpl() {
  GetOutput() = OutType{};
  return true;
}

bool TerekhovDTestTaskSEQ::RunImpl() {
  const InType &matrix = GetInput();
  if (matrix.empty()) {
    return false;
  }

  const std::size_t rows = matrix.size();
  const std::size_t cols = matrix[0].size();

  OutType result(cols, std::numeric_limits<int>::max());

  for (std::size_t row_idx = 0; row_idx < rows; ++row_idx) {
    for (std::size_t col_idx = 0; col_idx < cols; ++col_idx) {
      result[col_idx] = std::min(matrix[row_idx][col_idx], result[col_idx]);
    }
  }

  GetOutput() = result;
  return true;
}

bool TerekhovDTestTaskSEQ::PostProcessingImpl() {
  const InType &matrix = GetInput();
  if (matrix.empty()) {
    return false;
  }

  const OutType &out = GetOutput();
  return out.size() == matrix[0].size();
}

}  // namespace terekhov_d_a_test_task_processes
