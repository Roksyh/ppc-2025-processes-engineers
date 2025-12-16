#include <gtest/gtest.h>

#include <cstddef>
#include <utility>
#include <vector>

#include "Terekhov_D_Horizontal_matrix_vector/common/include/common.hpp"
#include "Terekhov_D_Horizontal_matrix_vector/mpi/include/ops_mpi.hpp"
#include "Terekhov_D_Horizontal_matrix_vector/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace terekhov_d_horizontal_matrix_vector {

class TerekhovDHorizontalMatrixVectorRunPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  static constexpr size_t kSize = 2000;

 protected:
  void SetUp() override {
    matrix_a_ = std::vector<std::vector<double>>(kSize, std::vector<double>(kSize));
    vector_b_ = std::vector<double>(kSize);

    for (size_t i = 0; i < kSize; ++i) {
      for (size_t j = 0; j < kSize; ++j) {
        matrix_a_[i][j] = static_cast<double>((i * kSize) + j) * 0.001;
      }
      vector_b_[i] = static_cast<double>(i) * 0.002;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return !output_data.empty();
  }

  InType GetTestInputData() final {
    return std::make_pair(matrix_a_, vector_b_);
  }

 private:
  std::vector<std::vector<double>> matrix_a_;
  std::vector<double> vector_b_;
};

TEST_P(TerekhovDHorizontalMatrixVectorRunPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, TerekhovDHorizontalMatrixVectorMPI, TerekhovDHorizontalMatrixVectorSEQ>(
        PPC_SETTINGS_Terekhov_D_Horizontal_matrix_vector);
const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = TerekhovDHorizontalMatrixVectorRunPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, TerekhovDHorizontalMatrixVectorRunPerfTests, kGtestValues, kPerfTestName);

}  // namespace terekhov_d_horizontal_matrix_vector
