#pragma once

#include <cstddef>
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

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  int world_rank_{0};
  int world_size_{1};
  int local_rows_{0};
  std::size_t cols_{0};
  std::vector<int> local_buffer_;
};  // commit

}  // namespace terekhov_d_a_test_task_processes
