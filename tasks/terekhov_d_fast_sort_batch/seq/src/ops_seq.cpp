#include "terekhov_d_fast_sort_batch/seq/include/ops_seq.hpp"

#include <algorithm>
#include <vector>

#include "terekhov_d_fast_sort_batch/common/include/common.hpp"

namespace terekhov_d_fast_sort_batch {

TerekhovDFastSortBatchSEQ::TerekhovDFastSortBatchSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>();
}

bool TerekhovDFastSortBatchSEQ::ValidationImpl() {
  return GetOutput().empty();
}

bool TerekhovDFastSortBatchSEQ::PreProcessingImpl() {
  return true;
}

bool TerekhovDFastSortBatchSEQ::RunImpl() {
  const auto &input_data = GetInput();
  if (input_data.empty()) {
    return true;
  }

  std::vector<int> sorted_buffer = input_data;
  std::sort(sorted_buffer.begin(), sorted_buffer.end());

  GetOutput() = std::move(sorted_buffer);
  return true;
}

bool TerekhovDFastSortBatchSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace terekhov_d_fast_sort_batch
