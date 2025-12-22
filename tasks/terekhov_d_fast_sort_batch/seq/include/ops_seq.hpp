#pragma once

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

  static void QuickSortWithBatcherMerge(std::vector<int> &arr, int left, int right);
  static int Partition(std::vector<int> &arr, int left, int right);
  static void BatcherOddEvenMerge(std::vector<int> &arr, int left, int mid, int right);
};

}  // namespace terekhov_d_fast_sort_batch
