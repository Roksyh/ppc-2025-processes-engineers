#include "Terekhov_D_Min_Column_Matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "Terekhov_D_Min_Column_Matrix/common/include/common.hpp"

namespace terekhov_d_a_test_task_processes {

TerekhovDTestTaskMPI::TerekhovDTestTaskMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType temp_in = in;
  GetInput().swap(temp_in);
  OutType temp_out;
  GetOutput().swap(temp_out);
}

static bool ValidateMatrixOnRoot(const terekhov_d_a_test_task_processes::InType &matrix) {
  if (matrix.empty()) {
    return false;
  }

  const std::size_t cols = matrix[0].size();
  if (cols == 0) {
    return false;
  }

  for (const auto &row : matrix) {
    if (row.size() != cols) {
      return false;
    }
  }

  return true;
}

bool TerekhovDTestTaskMPI::ValidationImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

  int valid_flag = 0;

  if (world_rank_ == 0) {
    valid_flag = ValidateMatrixOnRoot(GetInput()) ? 1 : 0;
  }

  MPI_Bcast(&valid_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return valid_flag == 1;
}

static std::vector<int> FlattenMatrix(const terekhov_d_a_test_task_processes::InType &matrix) {
  if (matrix.empty()) {
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

bool TerekhovDTestTaskMPI::PreProcessingImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

  int rows = 0;
  int cols = 0;

  if (world_rank_ == 0) {
    const InType &matrix = GetInput();
    rows = static_cast<int>(matrix.size());
    cols = matrix.empty() ? 0 : static_cast<int>(matrix[0].size());
  }

  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  cols_ = static_cast<std::size_t>(cols);

  if (rows == 0 || cols == 0) {
    local_rows_ = 0;
    local_buffer_.clear();
    GetOutput().clear();
    return true;
  }

  const int rows_per_process = rows / world_size_;
  const int remainder = rows % world_size_;

  std::vector<int> counts_rows(world_size_);
  std::vector<int> displs_rows(world_size_);

  int current_displ = 0;
  for (int rank = 0; rank < world_size_; ++rank) {
    const int rows_for_rank = rows_per_process + (rank < remainder ? 1 : 0);
    counts_rows[rank] = rows_for_rank;
    displs_rows[rank] = current_displ;
    current_displ += rows_for_rank;
  }

  local_rows_ = counts_rows[world_rank_];
  const int local_count_elems = local_rows_ * cols;
  local_buffer_.assign(local_count_elems, 0);

  if (world_rank_ == 0) {
    std::vector<int> flat = FlattenMatrix(GetInput());

    std::vector<int> counts_elems(world_size_);
    std::vector<int> displs_elems(world_size_);

    for (int rank = 0; rank < world_size_; ++rank) {
      counts_elems[rank] = counts_rows[rank] * cols;
      displs_elems[rank] = displs_rows[rank] * cols;
    }

    MPI_Scatterv(flat.data(), counts_elems.data(), displs_elems.data(), MPI_INT, local_buffer_.data(),
                 local_count_elems, MPI_INT, 0, MPI_COMM_WORLD);
  } else {
    MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT, local_buffer_.data(), local_count_elems, MPI_INT, 0,
                 MPI_COMM_WORLD);
  }

  GetOutput().clear();
  return true;
}

bool TerekhovDTestTaskMPI::RunImpl() {
  if (cols_ == 0) {
    GetOutput().clear();
    return true;
  }

  std::vector<int> local_min(cols_, std::numeric_limits<int>::max());

  if (local_rows_ > 0) {
    for (int row_idx = 0; row_idx < local_rows_; ++row_idx) {
      const int base_idx = row_idx * static_cast<int>(cols_);
      for (std::size_t col_idx = 0; col_idx < cols_; ++col_idx) {
        const int value = local_buffer_[base_idx + static_cast<int>(col_idx)];
        if (value < local_min[col_idx]) {
          local_min[col_idx] = value;
        }
      }
    }
  }

  std::vector<int> global_min(cols_, std::numeric_limits<int>::max());

  MPI_Allreduce(local_min.data(), global_min.data(), static_cast<int>(cols_), MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  GetOutput() = global_min;
  return true;
}

bool TerekhovDTestTaskMPI::PostProcessingImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);

  if (world_rank_ != 0) {
    return true;
  }

  const OutType &out = GetOutput();
  const InType &matrix = GetInput();

  if (matrix.empty()) {
    return out.empty();
  }

  return out.size() == matrix[0].size();
}

}  // namespace terekhov_d_a_test_task_processes
