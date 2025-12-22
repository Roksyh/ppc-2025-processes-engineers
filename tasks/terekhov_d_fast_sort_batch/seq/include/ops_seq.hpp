#pragma once

#include <utility>
#include <vector>

#include "task/include/task.hpp"
#include "terekhov_d_fast_sort_batch/common/include/common.hpp"

namespace terekhov_d_fast_sort_batch {

class TerekhovDFastSortBatchSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit TerekhovDFastSortBatchSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  static std::pair<int, int> PartitionSegment(std::vector<int> &arr, int start_idx, int end_idx);
};

}  // namespace terekhov_d_fast_sort_batch
