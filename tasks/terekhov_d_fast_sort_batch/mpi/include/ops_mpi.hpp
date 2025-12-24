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
  int process_rank_{};
  int process_count_{};

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void BroadcastArraySizes(std::size_t &actual_size, std::size_t &adjusted_size);
  void DistributeArrayData(const std::size_t &adjusted_size, const std::vector<int> &adjusted_input,
                           std::vector<int> &segment_sizes, std::vector<int> &segment_offsets,
                           std::vector<int> &local_segment) const;

  void GenerateComparatorPairs(std::vector<std::pair<int, int>> &comparator_pairs) const;
  std::pair<std::vector<int>, std::vector<int>> static SeparateOddEven(const std::vector<int> &elements);
  void static ConstructMergeNetwork(const std::vector<int> &upper_processes, const std::vector<int> &lower_processes,
                                    std::vector<std::pair<int, int>> &comparator_pairs);
  void static ConstructSortNetwork(const std::vector<int> &processes,
                                   std::vector<std::pair<int, int>> &comparator_pairs);

  void ExecuteComparatorPairs(const std::vector<int> &segment_sizes, std::vector<int> &local_segment,
                              const std::vector<std::pair<int, int>> &comparator_pairs) const;
  void static MergeDataSegments(const std::vector<int> &local_data, const std::vector<int> &partner_data,
                                std::vector<int> &result_buffer, bool take_smaller_values);
};

}  // namespace terekhov_d_fast_sort_batch
