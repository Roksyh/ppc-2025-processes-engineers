#pragma once

#include <utility>
#include <vector>

#include "Terekhov_D_Horizontal_matrix_vector/common/include/common.hpp"
#include "task/include/task.hpp"

namespace terekhov_d_horizontal_matrix_vector {

class TerekhovDHorizontalMatrixVectorSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit TerekhovDHorizontalMatrixVectorSEQ(InType in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  InType input_;
};

}  // namespace terekhov_d_horizontal_matrix_vector
