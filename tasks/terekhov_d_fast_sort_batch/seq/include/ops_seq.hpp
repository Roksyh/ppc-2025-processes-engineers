#pragma once

#include "terekhov_d_fast_sort_batch/common/include/common.hpp"
#include "task/include/task.hpp"

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

  InType data_;
};

}  // namespace terekhov_d_fast_sort_batch
