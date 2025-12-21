#include "terekhov_d_fast_sort_batch/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "terekhov_d_fast_sort_batch/common/include/common.hpp"

namespace terekhov_d_fast_sort_batch {

namespace {

void QuickSort(std::vector<int> *vec, int left, int right) {
  if (left >= right) return;
  
  auto &a = *vec;
  int pivot = a[(left + right) / 2];
  int i = left, j = right;
  
  while (i <= j) {
    while (a[i] < pivot) ++i;
    while (a[j] > pivot) --j;
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
  if (vec->empty()) return;
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

}  // namespace

TerekhovDFastSortBatchSEQ::TerekhovDFastSortBatchSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool TerekhovDFastSortBatchSEQ::ValidationImpl() {
  return true;
}

bool TerekhovDFastSortBatchSEQ::PreProcessingImpl() {
  data_ = GetInput();
  GetOutput().clear();
  return true;
}

bool TerekhovDFastSortBatchSEQ::RunImpl() {
  const std::size_t n = data_.size();
  if (n <= 1) {
    return true;
  }

  std::size_t blocks = 1;
  while ((blocks << 1U) <= n && (blocks << 1U) <= 16) {
    blocks <<= 1U;
  }

  std::vector<std::vector<int>> parts;
  parts.reserve(blocks);

  const std::size_t base = n / blocks;
  const std::size_t rem = n % blocks;

  std::size_t pos = 0;
  for (std::size_t block_index = 0; block_index < blocks; ++block_index) {
    const std::size_t sz = base + (block_index < rem ? 1 : 0);
    std::vector<int> chunk;
    chunk.reserve(sz);
    for (std::size_t i = 0; i < sz; ++i) {
      chunk.push_back(data_[pos++]);
    }
    QuickSort(&chunk);
    parts.push_back(std::move(chunk));
  }

  while (parts.size() > 1) {
    std::vector<std::vector<int>> next;
    next.reserve(parts.size() / 2);
    for (std::size_t i = 0; i < parts.size(); i += 2) {
      next.push_back(BatcherOddEvenMerge(parts[i], parts[i + 1]));
    }
    parts = std::move(next);
  }

  data_ = std::move(parts[0]);
  return true;
}

bool TerekhovDFastSortBatchSEQ::PostProcessingImpl() {
  GetOutput() = data_;
  return true;
}

}  // namespace terekhov_d_fast_sort_batch
