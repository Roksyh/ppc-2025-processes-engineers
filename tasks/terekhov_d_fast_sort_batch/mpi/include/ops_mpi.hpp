#pragma once

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
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static std::vector<int> GetProcessSegment(int proc_count, int rank, int arr_size);
  void BroadcastInputData(int rank);
  static void CombineAndSelect(std::vector<int> &local_vec, std::vector<int> &partner_vec, bool take_smaller);
  static void ProcessExchange(std::vector<int> &local_vec, int arr_size, int rank, int proc_count, int partner_offset);
  static void BatcherMergeProcedure(std::vector<int> &local_vec, std::vector<int> &segment, int arr_size, int rank,
                                    int proc_count);
  static void BatcherEvenStep(std::vector<int> &local_vec, std::vector<int> &segment, int arr_size, int rank,
                              int proc_count);
  static void BatcherOddStep(std::vector<int> &local_vec, std::vector<int> &segment, int arr_size, int rank,
                             int proc_count);
  static int GetLocalSize(int arr_size, int rank, int proc_count);
  static void SortLocalData(std::vector<int> &data);
  static std::pair<int, int> PartitionArray(std::vector<int> &arr, int start_idx, int end_idx);
  void DistributeFinalResult(int rank);
};

}  // namespace terekhov_d_fast_sort_batch
