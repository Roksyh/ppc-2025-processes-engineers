#pragma once

#include <cstddef>
#include <utility>
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
  int current_rank_{};
  int total_processes_{};

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void BroadcastArrayDimensions(size_t &initial_size, size_t &expanded_size);
  void DistributeArrayElements(const size_t &expanded_size, const std::vector<int> &expanded_input,
                               std::vector<int> &portion_sizes, std::vector<int> &portion_starts,
                               std::vector<int> &local_portion) const;

  void BuildComparatorSequence(std::vector<std::pair<int, int>> &comparator_list) const;
  std::pair<std::vector<int>, std::vector<int>> static SplitByPosition(const std::vector<int> &items);
  void static ConstructMergeStep(const std::vector<int> &top_group, const std::vector<int> &bottom_group,
                                 std::vector<std::pair<int, int>> &comparator_list);
  void static ConstructSortStep(const std::vector<int> &process_group,
                                std::vector<std::pair<int, int>> &comparator_list);

  void ExecuteComparators(const std::vector<int> &portion_sizes, std::vector<int> &local_portion,
                          const std::vector<std::pair<int, int>> &comparator_list) const;
  void static MergePortions(const std::vector<int> &local_data, const std::vector<int> &partner_data,
                            std::vector<int> &result_buffer, bool take_minimal);
};

}  // namespace terekhov_d_fast_sort_batch
