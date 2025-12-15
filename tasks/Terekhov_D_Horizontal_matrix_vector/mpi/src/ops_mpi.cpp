#include "terekhov_d_horizontal_matrix_vector/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include "terekhov_d_horizontal_matrix_vector/common/include/common.hpp"

namespace terekhov_d_horizontal_matrix_vector {

TerekhovDHorizontalMatrixVectorMPI::TerekhovDHorizontalMatrixVectorMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetOutput() = std::vector<double>();

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    matrix_A_ = in.first;
    vector_B_ = in.second;
  }
}

bool TerekhovDHorizontalMatrixVectorMPI::ValidationImpl() {
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);

  if (mpi_initialized == 0) {
    return false;
  }

  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  return size >= 1;
}

bool TerekhovDHorizontalMatrixVectorMPI::PreProcessingImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  rank_ = rank;
  world_size_ = size;
  GetOutput() = std::vector<double>();

  return true;
}

bool TerekhovDHorizontalMatrixVectorMPI::RunImpl() {
  if (world_size_ == 1) {
    return RunSequential();
  }

  int rows_a = 0;
  int cols_a = 0;
  int vector_size = 0;

  if (!PrepareAndValidateSizes(rows_a, cols_a, vector_size)) {
    return true;
  }

  std::vector<double> vector_flat(static_cast<size_t>(vector_size));
  PrepareAndBroadcastVector(vector_flat, vector_size);

  std::vector<int> my_row_indices;
  std::vector<double> local_a_flat;
  int local_rows = 0;
  DistributeMatrixAData(my_row_indices, local_a_flat, local_rows, rows_a, cols_a);

  std::vector<double> local_result_flat(static_cast<size_t>(local_rows), 0.0);
  ComputeLocalMultiplication(local_a_flat, vector_flat, local_result_flat, local_rows, cols_a);

  std::vector<double> final_result_flat;
  GatherResults(final_result_flat, my_row_indices, local_result_flat, local_rows, rows_a);

  GetOutput() = final_result_flat;

  return true;
}

bool TerekhovDHorizontalMatrixVectorMPI::RunSequential() {
  if (rank_ != 0) {
    return true;
  }

  const auto &matrix_a = matrix_A_;
  const auto &vector_b = vector_B_;

  if (matrix_a.empty() || vector_b.empty()) {
    GetOutput() = std::vector<double>();
    return true;
  }

  size_t rows_a = matrix_a.size();
  size_t cols_a = matrix_a[0].size();

  auto &output = GetOutput();
  output = std::vector<double>(rows_a, 0.0);

  for (size_t i = 0; i < rows_a; ++i) {
    for (size_t j = 0; j < cols_a; ++j) {
      output[i] += matrix_a[i][j] * vector_b[j];
    }
  }

  return true;
}

bool TerekhovDHorizontalMatrixVectorMPI::PrepareAndValidateSizes(int &rows_a, int &cols_a, int &vector_size) {
  if (rank_ == 0) {
    rows_a = static_cast<int>(matrix_A_.size());
    cols_a = rows_a > 0 ? static_cast<int>(matrix_A_[0].size()) : 0;
    vector_size = static_cast<int>(vector_B_.size());
  }

  std::array<int, 3> sizes = {rows_a, cols_a, vector_size};
  MPI_Bcast(sizes.data(), 3, MPI_INT, 0, MPI_COMM_WORLD);

  rows_a = sizes[0];
  cols_a = sizes[1];
  vector_size = sizes[2];

  if (cols_a != vector_size || rows_a == 0 || cols_a == 0 || vector_size == 0) {
    GetOutput() = std::vector<double>();
    return false;
  }

  return true;
}

void TerekhovDHorizontalMatrixVectorMPI::PrepareAndBroadcastVector(std::vector<double> &vector_flat, int vector_size) {
  if (rank_ == 0) {
    for (int i = 0; i < vector_size; ++i) {
      vector_flat[static_cast<size_t>(i)] = vector_B_[i];
    }
  }

  MPI_Bcast(vector_flat.data(), vector_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void TerekhovDHorizontalMatrixVectorMPI::FillLocalAFlat(const std::vector<int> &my_row_indices,
                                                        std::vector<double> &local_a_flat, int cols_a) {
  for (size_t idx = 0; idx < my_row_indices.size(); ++idx) {
    int global_row = my_row_indices[idx];
    for (int j = 0; j < cols_a; ++j) {
      local_a_flat[(idx * static_cast<size_t>(cols_a)) + static_cast<size_t>(j)] = matrix_A_[global_row][j];
    }
  }
}

void TerekhovDHorizontalMatrixVectorMPI::SendRowsToProcess(int dest, const std::vector<int> &dest_rows, int cols_a) {
  int dest_row_count = static_cast<int>(dest_rows.size());
  MPI_Send(&dest_row_count, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);

  if (dest_row_count > 0) {
    std::vector<int> rows_copy = dest_rows;
    MPI_Send(rows_copy.data(), dest_row_count, MPI_INT, dest, 1, MPI_COMM_WORLD);

    std::vector<double> buffer(static_cast<size_t>(dest_row_count) * static_cast<size_t>(cols_a));
    for (int idx = 0; idx < dest_row_count; ++idx) {
      int global_row = dest_rows[idx];
      for (int j = 0; j < cols_a; ++j) {
        buffer[(static_cast<size_t>(idx) * static_cast<size_t>(cols_a)) + static_cast<size_t>(j)] =
            matrix_A_[global_row][j];
      }
    }

    MPI_Send(buffer.data(), dest_row_count * cols_a, MPI_DOUBLE, dest, 2, MPI_COMM_WORLD);
  }
}

std::vector<int> TerekhovDHorizontalMatrixVectorMPI::GetRowsForProcess(int process_rank, int rows_a) const {
  std::vector<int> rows;
  for (int i = 0; i < rows_a; ++i) {
    if (i % world_size_ == process_rank) {
      rows.push_back(i);
    }
  }
  return rows;
}

void TerekhovDHorizontalMatrixVectorMPI::ReceiveRowsFromRoot(int &local_rows, std::vector<int> &my_row_indices,
                                                             std::vector<double> &local_a_flat, int cols_a) {
  MPI_Recv(&local_rows, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  if (local_rows > 0) {
    my_row_indices.resize(static_cast<size_t>(local_rows));
    local_a_flat.resize(static_cast<size_t>(local_rows) * static_cast<size_t>(cols_a));

    MPI_Recv(my_row_indices.data(), local_rows, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(local_a_flat.data(), local_rows * cols_a, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

void TerekhovDHorizontalMatrixVectorMPI::DistributeMatrixAData(std::vector<int> &my_row_indices,
                                                               std::vector<double> &local_a_flat, int &local_rows,
                                                               int rows_a, int cols_a) {
  local_rows = (rows_a / world_size_) + (rank_ < (rows_a % world_size_) ? 1 : 0);

  my_row_indices = GetRowsForProcess(rank_, rows_a);

  if (my_row_indices.size() != static_cast<size_t>(local_rows)) {
    local_rows = static_cast<int>(my_row_indices.size());
  }

  local_a_flat.resize(static_cast<size_t>(local_rows) * static_cast<size_t>(cols_a));

  if (rank_ == 0) {
    FillLocalAFlat(my_row_indices, local_a_flat, cols_a);

    for (int dest = 1; dest < world_size_; ++dest) {
      std::vector<int> dest_rows = GetRowsForProcess(dest, rows_a);
      SendRowsToProcess(dest, dest_rows, cols_a);
    }
  } else {
    ReceiveRowsFromRoot(local_rows, my_row_indices, local_a_flat, cols_a);
  }
}

void TerekhovDHorizontalMatrixVectorMPI::CollectLocalResults(const std::vector<int> &my_row_indices,
                                                             const std::vector<double> &local_result_flat,
                                                             std::vector<double> &final_result_flat) {
  for (size_t idx = 0; idx < my_row_indices.size(); ++idx) {
    int global_row = my_row_indices[idx];
    final_result_flat[static_cast<size_t>(global_row)] = local_result_flat[idx];
  }
}

void TerekhovDHorizontalMatrixVectorMPI::ReceiveResultsFromProcess(int src,
                                                                   std::vector<double> &final_result_flat) const {
  int rows_a = static_cast<int>(final_result_flat.size());
  std::vector<int> src_rows = GetRowsForProcess(src, rows_a);
  int src_row_count = static_cast<int>(src_rows.size());

  if (src_row_count > 0) {
    std::vector<double> buffer(static_cast<size_t>(src_row_count));
    MPI_Recv(buffer.data(), src_row_count, MPI_DOUBLE, src, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    for (int idx = 0; idx < src_row_count; ++idx) {
      int global_row = src_rows[idx];
      final_result_flat[static_cast<size_t>(global_row)] = buffer[static_cast<size_t>(idx)];
    }
  }
}

void TerekhovDHorizontalMatrixVectorMPI::SendLocalResults(const std::vector<double> &local_result_flat,
                                                          int local_rows) {
  if (local_rows > 0) {
    std::vector<double> data_copy = local_result_flat;
    MPI_Send(data_copy.data(), local_rows, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
  }
}

void TerekhovDHorizontalMatrixVectorMPI::GatherResults(std::vector<double> &final_result_flat,
                                                       const std::vector<int> &my_row_indices,
                                                       const std::vector<double> &local_result_flat, int local_rows,
                                                       int rows_a) const {
  if (rank_ == 0) {
    final_result_flat.resize(static_cast<size_t>(rows_a), 0.0);

    CollectLocalResults(my_row_indices, local_result_flat, final_result_flat);

    for (int src = 1; src < world_size_; ++src) {
      ReceiveResultsFromProcess(src, final_result_flat);
    }

    MPI_Bcast(final_result_flat.data(), rows_a, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  } else {
    SendLocalResults(local_result_flat, local_rows);

    final_result_flat.resize(static_cast<size_t>(rows_a));
    MPI_Bcast(final_result_flat.data(), rows_a, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
}

void TerekhovDHorizontalMatrixVectorMPI::ComputeLocalMultiplication(const std::vector<double> &local_a_flat,
                                                                    const std::vector<double> &vector_flat,
                                                                    std::vector<double> &local_result_flat,
                                                                    int local_rows, int cols_a) {
  for (int i = 0; i < local_rows; ++i) {
    const double *a_row = &local_a_flat[static_cast<size_t>(i) * static_cast<size_t>(cols_a)];
    double sum = 0.0;

    for (int j = 0; j < cols_a; ++j) {
      sum += a_row[j] * vector_flat[static_cast<size_t>(j)];
    }

    local_result_flat[static_cast<size_t>(i)] = sum;
  }
}

bool TerekhovDHorizontalMatrixVectorMPI::PostProcessingImpl() {
  return true;
}

}  // namespace terekhov_d_horizontal_matrix_vector
