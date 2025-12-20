#pragma once

#include <utility>
#include <vector>

#include "Terekhov_D_Min_Column_Matrix/common/include/common.hpp"
#include "task/include/task.hpp"

namespace terekhov_d_a_test_task_processes {

class TerekhovDTestTaskMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit TerekhovDTestTaskMPI(const InType &in);

  static std::vector<int> FlattenMatrix(const std::vector<std::vector<int>> &matrix);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static std::pair<int, int> PrepareDimensions(const std::vector<std::vector<int>> &matrix, int rank);
  static std::pair<int, int> CalculateProcessInfo(int total_rows, int size, int rank);
  static std::vector<int> ScatterData(const std::vector<std::vector<int>> &matrix, int total_rows, int total_cols,
                                      int size, int rank, int my_rows);
  static std::vector<int> ComputeLocalColumnMinima(const std::vector<int> &local_data, int my_rows, int total_cols);
  static std::vector<int> ReduceColumnResults(const std::vector<int> &local_minima, int total_cols);
};

}  // namespace terekhov_d_a_test_task_processes
