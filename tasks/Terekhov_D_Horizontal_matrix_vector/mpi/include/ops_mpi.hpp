#pragma once

#include <vector>

#include "Terekhov_D_Horizontal_matrix_vector/common/include/common.hpp"
#include "task/include/task.hpp"

namespace terekhov_d_horizontal_matrix_vector {

class TerekhovDHorizontalMatrixVectorMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit TerekhovDHorizontalMatrixVectorMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  bool RunSequential();

  bool PrepareAndValidateSizes(int &rows_a, int &cols_a, int &vector_size);
  void PrepareAndBroadcastVector(std::vector<double> &vector_flat, int vector_size);
  void DistributeMatrixAData(std::vector<int> &my_row_indices, std::vector<double> &local_a_flat, int &local_rows,
                             int rows_a, int cols_a);
  static void ComputeLocalMultiplication(const std::vector<double> &local_a_flat,
                                         const std::vector<double> &vector_flat, std::vector<double> &local_result_flat,
                                         int local_rows, int cols_a);
  void GatherResults(std::vector<double> &final_result_flat, const std::vector<int> &my_row_indices,
                     const std::vector<double> &local_result_flat, int local_rows, int rows_a) const;

  void FillLocalAFlat(const std::vector<int> &my_row_indices, std::vector<double> &local_a_flat, int cols_a);
  void SendRowsToProcess(int dest, const std::vector<int> &dest_rows, int cols_a);
  [[nodiscard]] std::vector<int> GetRowsForProcess(int process_rank, int rows_a) const;
  static void ReceiveRowsFromRoot(int &local_rows, std::vector<int> &my_row_indices, std::vector<double> &local_a_flat,
                                  int cols_a);

  static void CollectLocalResults(const std::vector<int> &my_row_indices, const std::vector<double> &local_result_flat,
                                  std::vector<double> &final_result_flat);
  void ReceiveResultsFromProcess(int src, std::vector<double> &final_result_flat) const;
  static void SendLocalResults(const std::vector<double> &local_result_flat, int local_rows);

  std::vector<std::vector<double>> matrix_A_;
  std::vector<double> vector_B_;
  std::vector<double> result_vector_;
  int rank_ = 0;
  int world_size_ = 1;
};

}  // namespace terekhov_d_horizontal_matrix_vector
