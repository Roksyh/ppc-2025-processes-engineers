#pragma once

#include <vector>

#include "task/include/task.hpp"
#include "terekhov_d_fast_sort_batch/common/include/common.hpp"

namespace terekhov_d_fast_sort_batch {

class TerekhovDFastSortBatchMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit TerekhovDFastSortBatchMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static int Partition(std::vector<int> &arr, int left, int right);
  static void BatcherOddEvenMerge(std::vector<int> &arr, int left, int mid, int right);
  static void QuickSortWithBatcherMerge(std::vector<int> &arr, int left, int right);

  void ParallelQuickSort();
  std::vector<int> DistributeData(int world_size, int world_rank);

  struct DistributionInfo {
    std::vector<int> send_counts;
    std::vector<int> displacements;
    int local_size = 0;
  };

  static DistributionInfo PrepareDistributionInfo(int total_size, int world_size, int world_rank);
};

}  // namespace terekhov_d_fast_sort_batch
