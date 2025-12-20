#include "Terekhov_D_Min_Column_Matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <array>
#include <climits>
#include <cstddef>
#include <ranges>
#include <utility>
#include <vector>

#include "Terekhov_D_Min_Column_Matrix/common/include/common.hpp"

namespace terekhov_d_a_test_task_processes {

TerekhovDTestTaskMPI::TerekhovDTestTaskMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  if (!in.empty()) {
    GetInput() = in;
  } else {
    GetInput() = InType{};
  }

  GetOutput() = OutType{};
}

bool TerekhovDTestTaskMPI::ValidationImpl() {
  const auto &input = GetInput();
  if (input.empty()) {
    return false;
  }

  const std::size_t cols = input[0].size();
  if (cols == 0) {
    return false;
  }

  return std::ranges::all_of(input, [cols](const auto &row) { return row.size() == cols; });
}

bool TerekhovDTestTaskMPI::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

std::vector<int> TerekhovDTestTaskMPI::FlattenMatrix(const std::vector<std::vector<int>> &matrix) {
  if (matrix.empty() || matrix[0].empty()) {
    return {};
  }

  const std::size_t rows = matrix.size();
  const std::size_t cols = matrix[0].size();

  std::vector<int> flat;
  flat.reserve(rows * cols);

  for (const auto &row : matrix) {
    flat.insert(flat.end(), row.begin(), row.end());
  }

  return flat;
}

std::pair<int, int> TerekhovDTestTaskMPI::PrepareDimensions(const std::vector<std::vector<int>> &matrix, int rank) {
  int total_rows = 0;
  int total_cols = 0;

  if (rank == 0) {
    total_rows = static_cast<int>(matrix.size());
    total_cols = (total_rows > 0) ? static_cast<int>(matrix[0].size()) : 0;
  }

  std::array<int, 2> dimensions = {total_rows, total_cols};
  MPI_Bcast(dimensions.data(), 2, MPI_INT, 0, MPI_COMM_WORLD);

  return {dimensions[0], dimensions[1]};
}

static std::vector<int> ScatterMatrixData(const std::vector<std::vector<int>> &matrix, int total_rows, int total_cols,
                                          int size, int rank, int my_rows) {
  if (size <= 0 || total_rows <= 0 || total_cols <= 0) {
    return std::vector<int>();
  }

  const std::size_t local_size = static_cast<std::size_t>(my_rows) * static_cast<std::size_t>(total_cols);
  std::vector<int> local_data(local_size, 0);

  if (rank == 0) {
    std::vector<int> flat_matrix = TerekhovDTestTaskMPI::FlattenMatrix(matrix);

    const int rows_per_process = total_rows / size;
    const int remainder = total_rows % size;

    std::vector<int> send_counts(size);
    std::vector<int> displacements(size);

    int current_displ = 0;
    for (int i = 0; i < size; ++i) {
      const int rows_for_i = rows_per_process + (i < remainder ? 1 : 0);
      send_counts[i] = rows_for_i * total_cols;
      displacements[i] = current_displ;
      current_displ += rows_for_i * total_cols;
    }

    MPI_Scatterv(flat_matrix.data(), send_counts.data(), displacements.data(), MPI_INT, local_data.data(),
                 my_rows * total_cols, MPI_INT, 0, MPI_COMM_WORLD);
  } else {
    MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT, local_data.data(), my_rows * total_cols, MPI_INT, 0,
                 MPI_COMM_WORLD);
  }

  return local_data;
}

static std::vector<int> ComputeLocalColumnMinima(const std::vector<int> &local_data, int my_rows, int total_cols) {
  std::vector<int> local_minima(static_cast<std::size_t>(total_cols), INT_MAX);

  if (my_rows > 0 && !local_data.empty()) {
    for (int i = 0; i < my_rows; ++i) {
      for (int j = 0; j < total_cols; ++j) {
        const int idx = (i * total_cols) + j;
        const int val = local_data[static_cast<std::size_t>(idx)];
        local_minima[static_cast<std::size_t>(j)] = std::min(val, local_minima[static_cast<std::size_t>(j)]);
      }
    }
  }

  return local_minima;
}

static std::vector<int> ReduceGlobalMinima(const std::vector<int> &local_minima, int total_cols) {
  if (total_cols <= 0) {
    return std::vector<int>();
  }

  std::vector<int> global_minima(static_cast<std::size_t>(total_cols), INT_MAX);

  MPI_Allreduce(local_minima.data(), global_minima.data(), total_cols, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  return global_minima;
}

bool TerekhovDTestTaskMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &matrix = GetInput();

  auto [total_rows, total_cols] = PrepareDimensions(matrix, rank);

  if (total_rows == 0 || total_cols == 0) {
    GetOutput() = OutType{};
    return true;
  }

  const int rows_per_process = total_rows / size;
  const int remainder = total_rows % size;
  const int my_rows = rows_per_process + (rank < remainder ? 1 : 0);

  std::vector<int> local_data = ScatterMatrixData(matrix, total_rows, total_cols, size, rank, my_rows);

  std::vector<int> local_minima = ComputeLocalColumnMinima(local_data, my_rows, total_cols);

  std::vector<int> global_minima = ReduceGlobalMinima(local_minima, total_cols);

  GetOutput() = std::move(global_minima);

  return true;
}

bool TerekhovDTestTaskMPI::PostProcessingImpl() {
  return true;
}

}  // namespace terekhov_d_a_test_task_processes
