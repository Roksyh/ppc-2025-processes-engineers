#pragma once

#include "Terekhov_D_Min_Column_Matrix/common/include/common.hpp"
#include "task/include/task.hpp"

namespace terekhov_d_a_test_task_processes {

class TerekhovDTestTaskSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit TerekhovDTestTaskSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace terekhov_d_a_test_task_processes
