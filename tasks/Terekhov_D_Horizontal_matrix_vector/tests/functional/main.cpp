#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "terekhov_d_horizontal_matrix_vector/common/include/common.hpp"
#include "terekhov_d_horizontal_matrix_vector/mpi/include/ops_mpi.hpp"
#include "terekhov_d_horizontal_matrix_vector/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace terekhov_d_horizontal_matrix_vector {

class TerekhovDHorizontalMatrixVectorFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_matrix_vector";
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    matrixA_ = std::get<1>(params);
    vectorB_ = std::get<2>(params);
    expected_ = std::get<3>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_.size()) {
      return false;
    }

    const double tolerance = 1e-10;

    for (size_t i = 0; i < expected_.size(); i++) {
      if (std::abs(output_data[i] - expected_[i]) > tolerance) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return std::make_pair(matrixA_, vectorB_);
  }

 private:
  std::vector<std::vector<double>> matrixA_;
  std::vector<double> vectorB_;
  std::vector<double> expected_;
};

namespace {

TEST_P(TerekhovDHorizontalMatrixVectorFuncTests, FunctionalTests) {
  ExecuteTest(GetParam());
}

TEST_P(TerekhovDHorizontalMatrixVectorFuncTests, CoverageTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 22> kFunctionalTests = {
    std::make_tuple(1, std::vector<std::vector<double>>{{1, 2}, {3, 4}}, std::vector<double>{5, 6},
                    std::vector<double>{17, 39}),

    std::make_tuple(2, std::vector<std::vector<double>>{{1, 0}, {0, 1}}, std::vector<double>{1, 2},
                    std::vector<double>{1, 2}),

    std::make_tuple(3, std::vector<std::vector<double>>{{1, 1}, {1, 1}}, std::vector<double>{1, 1},
                    std::vector<double>{2, 2}),

    std::make_tuple(4, std::vector<std::vector<double>>{{2, 0}, {0, 3}}, std::vector<double>{4, 5},
                    std::vector<double>{8, 15}),

    std::make_tuple(5, std::vector<std::vector<double>>{{1, 2, 3}}, std::vector<double>{4, 5, 6},
                    std::vector<double>{32}),

    std::make_tuple(6, std::vector<std::vector<double>>{{1}, {2}, {3}}, std::vector<double>{4},
                    std::vector<double>{4, 8, 12}),

    std::make_tuple(7, std::vector<std::vector<double>>{{1, 2, 3}, {4, 5, 6}}, std::vector<double>{7, 8, 9},
                    std::vector<double>{50, 122}),

    std::make_tuple(8, std::vector<std::vector<double>>{{2}}, std::vector<double>{3}, std::vector<double>{6}),

    std::make_tuple(9, std::vector<std::vector<double>>{{0, 0}, {0, 0}}, std::vector<double>{1, 2},
                    std::vector<double>{0, 0}),

    std::make_tuple(10, std::vector<std::vector<double>>{{1, 2}, {3, 4}}, std::vector<double>{0, 0},
                    std::vector<double>{0, 0}),

    std::make_tuple(11, std::vector<std::vector<double>>{{1, 2}, {3, 4}, {5, 6}}, std::vector<double>{7, 8},
                    std::vector<double>{23, 53, 83}),

    std::make_tuple(12, std::vector<std::vector<double>>{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, std::vector<double>{1, 2, 3},
                    std::vector<double>{1, 2, 3}),

    std::make_tuple(13, std::vector<std::vector<double>>{{0.5, 0.5}, {0.5, 0.5}}, std::vector<double>{2, 4},
                    std::vector<double>{3, 3}),

    std::make_tuple(14, std::vector<std::vector<double>>{{1, 1}, {1, 1}, {1, 1}}, std::vector<double>{1, 1},
                    std::vector<double>{2, 2, 2}),

    std::make_tuple(15, std::vector<std::vector<double>>{{2, 4}, {6, 8}}, std::vector<double>{1, 1},
                    std::vector<double>{6, 14}),

    std::make_tuple(16, std::vector<std::vector<double>>{{1, 2}, {3, 4}, {5, 6}}, std::vector<double>{7, 8},
                    std::vector<double>{23, 53, 83}),

    std::make_tuple(17, std::vector<std::vector<double>>{{0.1, 0.2}, {0.3, 0.4}}, std::vector<double>{5, 6},
                    std::vector<double>{1.7, 3.9}),

    std::make_tuple(18, std::vector<std::vector<double>>{{2, 2}, {2, 2}}, std::vector<double>{2, 2},
                    std::vector<double>{8, 8}),

    std::make_tuple(19, std::vector<std::vector<double>>{{3}}, std::vector<double>{4}, std::vector<double>{12}),

    std::make_tuple(20, std::vector<std::vector<double>>{{1, 0}, {0, 0}}, std::vector<double>{0, 1},
                    std::vector<double>{0, 0}),

    std::make_tuple(21, std::vector<std::vector<double>>{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, std::vector<double>{5, 6, 7},
                    std::vector<double>{5, 6, 7}),

    std::make_tuple(22, std::vector<std::vector<double>>{{1, 2}, {3, 4}}, std::vector<double>{0, 0},
                    std::vector<double>{0, 0})};

const std::array<TestType, 10> kCoverageTests = {
    std::make_tuple(23, std::vector<std::vector<double>>{{1}}, std::vector<double>{1}, std::vector<double>{1}),

    std::make_tuple(24, std::vector<std::vector<double>>{{0}}, std::vector<double>{0}, std::vector<double>{0}),

    std::make_tuple(25, std::vector<std::vector<double>>{{1, 2}, {3, 4}}, std::vector<double>{1, 0},
                    std::vector<double>{1, 3}),

    std::make_tuple(26, std::vector<std::vector<double>>{{1, 1}, {1, 1}}, std::vector<double>{2, 2},
                    std::vector<double>{4, 4}),

    std::make_tuple(27, std::vector<std::vector<double>>{{0.5}}, std::vector<double>{2}, std::vector<double>{1}),

    std::make_tuple(28, std::vector<std::vector<double>>{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, std::vector<double>{5, 6, 7},
                    std::vector<double>{5, 6, 7}),

    std::make_tuple(29, std::vector<std::vector<double>>{{2, 4}, {6, 8}}, std::vector<double>{0.5, 0.5},
                    std::vector<double>{3, 7}),

    std::make_tuple(30, std::vector<std::vector<double>>{{1, 2, 3}, {4, 5, 6}}, std::vector<double>{7, 8, 9},
                    std::vector<double>{50, 122}),

    std::make_tuple(31, std::vector<std::vector<double>>{{1}, {2}, {3}}, std::vector<double>{4},
                    std::vector<double>{4, 8, 12}),

    std::make_tuple(32, std::vector<std::vector<double>>{{1, 2, 3}}, std::vector<double>{4, 5, 6},
                    std::vector<double>{32})};

const auto kFunctionalTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<terekhov_d_horizontal_matrix_vector::TerekhovDHorizontalMatrixVectorMPI, InType>(
        kFunctionalTests, PPC_SETTINGS_Terekhov_D_Horizontal_matrix_vector),
    ppc::util::AddFuncTask<terekhov_d_horizontal_matrix_vector::TerekhovDHorizontalMatrixVectorSEQ, InType>(
        kFunctionalTests, PPC_SETTINGS_Terekhov_D_Horizontal_matrix_vector));

const auto kCoverageTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<terekhov_d_horizontal_matrix_vector::TerekhovDHorizontalMatrixVectorMPI, InType>(
        kCoverageTests, PPC_SETTINGS_Terekhov_D_Horizontal_matrix_vector),
    ppc::util::AddFuncTask<terekhov_d_horizontal_matrix_vector::TerekhovDHorizontalMatrixVectorSEQ, InType>(
        kCoverageTests, PPC_SETTINGS_Terekhov_D_Horizontal_matrix_vector));

inline const auto kFunctionalGtestValues = ppc::util::ExpandToValues(kFunctionalTasksList);
inline const auto kCoverageGtestValues = ppc::util::ExpandToValues(kCoverageTasksList);

inline const auto kPerfTestName =
    TerekhovDHorizontalMatrixVectorFuncTests::PrintFuncTestName<TerekhovDHorizontalMatrixVectorFuncTests>;

INSTANTIATE_TEST_SUITE_P(Functional, TerekhovDHorizontalMatrixVectorFuncTests, kFunctionalGtestValues, kPerfTestName);
INSTANTIATE_TEST_SUITE_P(Coverage, TerekhovDHorizontalMatrixVectorFuncTests, kCoverageGtestValues, kPerfTestName);

}  // namespace

}  // namespace terekhov_d_horizontal_matrix_vector
