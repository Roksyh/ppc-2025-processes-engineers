#include "terekhov_d_fast_sort_batch/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

namespace terekhov_d_fast_sort_batch {

namespace {

void QuickSort(std::vector<int> *vec, int left, int right) {
  if (left >= right) {
    return;
  }

  auto &a = *vec;
  int pivot = a[(left + right) / 2];
  int i = left, j = right;

  while (i <= j) {
    while (a[i] < pivot) {
      ++i;
    }
    while (a[j] > pivot) {
      --j;
    }
    if (i <= j) {
      std::swap(a[i], a[j]);
      ++i;
      --j;
    }
  }

  QuickSort(vec, left, j);
  QuickSort(vec, i, right);
}

void QuickSort(std::vector<int> *vec) {
  if (vec->empty()) {
    return;
  }
  QuickSort(vec, 0, static_cast<int>(vec->size()) - 1);
}

struct Elem {
  int val{};
  bool pad{false};

  Elem(int value, bool padding) : val(value), pad(padding) {}
};

inline bool Greater(const Elem &lhs, const Elem &rhs) {
  if (lhs.pad != rhs.pad) {
    return lhs.pad && !rhs.pad;
  }
  return lhs.val > rhs.val;
}

inline void CompareExchange(std::vector<Elem> *arr, std::size_t i, std::size_t j) {
  if (Greater((*arr)[i], (*arr)[j])) {
    std::swap((*arr)[i], (*arr)[j]);
  }
}

std::size_t NextPow2(std::size_t x) {
  std::size_t p = 1;
  while (p < x) {
    p <<= 1U;
  }
  return p;
}

void OddEvenMergeStep(std::vector<Elem> *arr, std::size_t k, std::size_t j) {
  const std::size_t n = arr->size();
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t ixj = i ^ j;
    if (ixj > i) {
      if ((i & k) == 0) {
        CompareExchange(arr, i, ixj);
      } else {
        CompareExchange(arr, ixj, i);
      }
    }
  }
}

void OddEvenMergeNetwork(std::vector<Elem> *arr) {
  const std::size_t n = arr->size();
  if (n <= 1) {
    return;
  }

  for (std::size_t k = 2; k <= n; k <<= 1U) {
    for (std::size_t j = (k >> 1U); j > 0; j >>= 1U) {
      OddEvenMergeStep(arr, k, j);
    }
  }
}

std::vector<int> BatcherOddEvenMerge(const std::vector<int> &a, const std::vector<int> &b) {
  const std::size_t need = a.size() + b.size();
  const std::size_t half = NextPow2((a.size() > b.size()) ? a.size() : b.size());

  std::vector<Elem> buffer;
  buffer.reserve(2 * half);

  for (std::size_t i = 0; i < half; ++i) {
    if (i < a.size()) {
      buffer.emplace_back(a[i], false);
    } else {
      buffer.emplace_back(0, true);
    }
  }
  for (std::size_t i = 0; i < half; ++i) {
    if (i < b.size()) {
      buffer.emplace_back(b[i], false);
    } else {
      buffer.emplace_back(0, true);
    }
  }

  OddEvenMergeNetwork(&buffer);

  std::vector<int> out;
  out.reserve(need);
  for (const auto &elem : buffer) {
    if (!elem.pad) {
      out.push_back(elem.val);
    }
    if (out.size() == need) {
      break;
    }
  }
  return out;
}

std::vector<int> RecvVector(int src, int tag_base, MPI_Comm comm) {
  int sz = 0;
  MPI_Status status{};
  MPI_Recv(&sz, 1, MPI_INT, src, tag_base, comm, &status);

  std::vector<int> v(static_cast<std::size_t>(sz));
  if (sz > 0) {
    MPI_Recv(v.data(), sz, MPI_INT, src, tag_base + 1, comm, &status);
  }
  return v;
}

void SendVector(int dst, int tag_base, const std::vector<int> &v, MPI_Comm comm) {
  const int sz = static_cast<int>(v.size());
  MPI_Send(&sz, 1, MPI_INT, dst, tag_base, comm);
  if (sz > 0) {
    MPI_Send(v.data(), sz, MPI_INT, dst, tag_base + 1, comm);
  }
}

}  // namespace

TerekhovDFastSortBatchMPI::TerekhovDFastSortBatchMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool TerekhovDFastSortBatchMPI::ValidationImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
  return true;
}

bool TerekhovDFastSortBatchMPI::PreProcessingImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

  int global_size = 0;
  if (world_rank_ == 0) {
    global_size = static_cast<int>(GetInput().size());
  }
  MPI_Bcast(&global_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  counts_.assign(world_size_, 0);
  displs_.assign(world_size_, 0);

  const int base = (world_size_ > 0) ? (global_size / world_size_) : 0;
  const int rem = (world_size_ > 0) ? (global_size % world_size_) : 0;

  for (int rank_index = 0; rank_index < world_size_; ++rank_index) {
    counts_[rank_index] = base + ((rank_index < rem) ? 1 : 0);
  }
  for (int rank_index = 1; rank_index < world_size_; ++rank_index) {
    displs_[rank_index] = displs_[rank_index - 1] + counts_[rank_index - 1];
  }

  local_.assign(static_cast<std::size_t>(counts_[world_rank_]), 0);

  int *send_buf = nullptr;
  if (world_rank_ == 0 && global_size > 0) {
    send_buf = GetInput().data();
  }

  MPI_Scatterv(send_buf, counts_.data(), displs_.data(), MPI_INT, local_.data(), counts_[world_rank_], MPI_INT, 0,
               MPI_COMM_WORLD);

  GetOutput().clear();
  return true;
}

bool TerekhovDFastSortBatchMPI::RunImpl() {
  QuickSort(&local_);

  for (int step = 1; step < world_size_; step <<= 1) {
    if ((world_rank_ % (2 * step)) == 0) {
      const int partner = world_rank_ + step;
      if (partner < world_size_) {
        std::vector<int> other = RecvVector(partner, 1000 + step, MPI_COMM_WORLD);
        local_ = BatcherOddEvenMerge(local_, other);
      }
    } else {
      const int partner = world_rank_ - step;
      SendVector(partner, 1000 + step, local_, MPI_COMM_WORLD);
      break;
    }
  }
  return true;
}

bool TerekhovDFastSortBatchMPI::PostProcessingImpl() {
  int out_size = 0;
  if (world_rank_ == 0) {
    GetOutput() = local_;
    out_size = static_cast<int>(GetOutput().size());
  }

  MPI_Bcast(&out_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (world_rank_ != 0) {
    GetOutput().assign(static_cast<std::size_t>(out_size), 0);
  }

  if (out_size > 0) {
    MPI_Bcast(GetOutput().data(), out_size, MPI_INT, 0, MPI_COMM_WORLD);
  }
  return true;
}

}  // namespace terekhov_d_fast_sort_batch
