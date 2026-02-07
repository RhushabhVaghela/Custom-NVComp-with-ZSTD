// ============================================================================
// cuda_zstd_memory_pool_simple.cu - Simplified Memory Pool Manager
// Implementation
// ============================================================================

#include "cuda_zstd_memory_pool.h"
#include "cuda_zstd_stacktrace.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <queue>
#include <random>
#include <thread>
#include <unordered_map>

// Logging macro for fallback events

namespace cuda_zstd {
namespace memory {

// ============================================================================
// Static Pool Sizes Definition
// ============================================================================

constexpr size_t MemoryPoolManager::POOL_SIZES[NUM_POOL_SIZES];

// ============================================================================
// Constructor and Destructor
// ============================================================================

MemoryPoolManager::MemoryPoolManager(bool enable_defrag)
    : defrag_enabled_(enable_defrag) {
  // Initialize fallback configuration
  fallback_config_.enable_host_memory_fallback = true;
  fallback_config_.enable_progressive_degradation = true;
  fallback_config_.enable_chunk_reduction = true;
  fallback_config_.enable_rollback_protection = true;
  fallback_config_.emergency_threshold_mb = 100;
  fallback_config_.host_memory_limit_mb = 1024;
  fallback_config_.degradation_factor = 0.5f;
  fallback_config_.max_retry_attempts = 3;

  // Pre-allocate vector capacity for each pool to avoid reallocation
  for (int i = 0; i < NUM_POOL_SIZES; ++i) {
    pools_[i].reserve(16); // Start with capacity for 16 entries per pool
  }

  // Initialize pressure tracking
  last_pressure_update_ = std::chrono::steady_clock::now();
}

MemoryPoolManager::~MemoryPoolManager() { clear(); }

// ============================================================================
// Helper Functions
// ============================================================================

int MemoryPoolManager::get_pool_index(size_t size) const {
  // Find the appropriate pool index for the requested size
  for (int i = 0; i < NUM_POOL_SIZES; ++i) {
    if (size <= POOL_SIZES[i]) {
      return i;
    }
  }
  return -1; // Size exceeds largest pool
}

size_t MemoryPoolManager::round_up_to_pool_size(size_t size) const {
  for (int i = 0; i < NUM_POOL_SIZES; ++i) {
    if (size <= POOL_SIZES[i]) {
      return POOL_SIZES[i];
    }
  }
  // Round up to nearest MB for very large allocations
  return ((size + SIZE_1MB - 1) / SIZE_1MB) * SIZE_1MB;
}

void MemoryPoolManager::update_peak_usage(size_t current_usage) {
  size_t current_peak = peak_memory_usage_.load(std::memory_order_relaxed);
  while (current_usage > current_peak) {
    if (peak_memory_usage_.compare_exchange_weak(current_peak, current_usage,
                                                 std::memory_order_relaxed)) {
      break;
    }
  }
}

void *MemoryPoolManager::allocate_from_cuda(size_t size) {
  void *ptr = nullptr;

  // CRITICAL: In emergency mode, do not allocate GPU memory
  // This forces fallback to host memory as intended
  if (is_emergency_mode()) {
    allocation_failures_.fetch_add(1, std::memory_order_relaxed);
    return nullptr;
  }

  // Check against max pool size (soft limit)
  if (current_memory_usage_.load(std::memory_order_relaxed) + size >
      max_pool_size_) {
    allocation_failures_.fetch_add(1, std::memory_order_relaxed);
    return nullptr;
  }

  cudaError_t err = cudaMalloc(&ptr, size);

  if (err != cudaSuccess) {
    // Log CUDA allocation failure for monitoring
    allocation_failures_.fetch_add(1, std::memory_order_relaxed);
    return nullptr;
  }

  size_t new_usage =
      current_memory_usage_.fetch_add(size, std::memory_order_relaxed) + size;
  update_peak_usage(new_usage);

  return ptr;
}

void *MemoryPoolManager::allocate_from_host(size_t size) {
  // Check host memory limit
  size_t current_host_usage =
      host_memory_usage_.load(std::memory_order_relaxed);
  if (current_host_usage + size >
      fallback_config_.host_memory_limit_mb * 1024 * 1024) {
    return nullptr;
  }

  // Use regular malloc instead of cudaMallocHost for compatibility
  // Tests expect to free with free(), not cudaFreeHost()
  void *ptr = cuda_zstd::util::debug_alloc(size);
  if (ptr) {
    host_memory_usage_.fetch_add(size, std::memory_order_relaxed);
    host_memory_allocations_.fetch_add(1, std::memory_order_relaxed);
  }

  return ptr;
}

// Attempt to lock the provided pool indices in ascending order. This helper
// will optionally use the same environment-driven timeout as other pool
// locking operations and return false if the deadline is reached.
bool MemoryPoolManager::lock_pools_ordered(
    const std::vector<int> &indices, unsigned timeout_ms,
    std::vector<std::unique_lock<std::timed_mutex>> &locks) const {
  // Make a local sorted copy of indices (remove duplicates)
  std::vector<int> idxs = indices;
  std::sort(idxs.begin(), idxs.end());
  idxs.erase(std::unique(idxs.begin(), idxs.end()), idxs.end());

  auto deadline = (timeout_ms == 0)
                      ? std::chrono::steady_clock::time_point::max()
                      : std::chrono::steady_clock::now() +
                            std::chrono::milliseconds(timeout_ms);

  // Attempt to lock each mutex in order, with retries until the deadline.
  while (true) {
    locks.clear();
    // CRITICAL FIX: Pre-reserve capacity to prevent reallocation during
    // emplace_back. Reallocation invalidates unique_lock objects causing UB.
    locks.reserve(idxs.size());
    bool ok = true;
    for (int id : idxs) {
      locks.emplace_back(pool_mutexes_[id], std::defer_lock);
      unsigned long attempt_timeout = 0;
      if (timeout_ms > 0) {
        auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
          ok = false;
          break;
        }
        attempt_timeout = static_cast<unsigned long>(
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline -
                                                                  now)
                .count());
      }

      if (attempt_timeout == 0) {
        locks.back().lock();
      } else {
        if (!locks.back().try_lock_for(
                std::chrono::milliseconds(attempt_timeout))) {
          ok = false;
          break;
        }
      }
    }

    if (ok)
      return true;

    // Failed to acquire; release and retry if there's time
    for (auto &l : locks) {
      if (l.owns_lock())
        l.unlock();
    }
    locks.clear();

    if (timeout_ms == 0) {
      // For infinite locks we never reach here since blocking locks were used
      return false;
    }

    if (std::chrono::steady_clock::now() >= deadline)
      return false;

    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

// Read a configurable lock timeout from the environment. A value of 0
// (default) means block indefinitely (preserve previous semantics).
static unsigned get_pool_lock_timeout_ms() {
  const char *env = getenv("CUDA_ZSTD_POOL_LOCK_TIMEOUT_MS");
  if (!env)
    return 0;
  try {
    return static_cast<unsigned>(std::stoul(env));
  } catch (...) {
    return 0;
  }
}

// Determine if pointer corresponds to GPU device memory. This is useful to
// detect host-fallback pointers that the pool may have returned. It first
// checks the pool entries (host fallback flag) and then queries CUDA for
// pointer attributes (device vs host memory).
bool MemoryPoolManager::is_device_pointer(void *ptr) const {
  if (!ptr)
    return false;

  // Check pool entries quickly for host fallback
  for (int i = 0; i < NUM_POOL_SIZES; ++i) {
    std::unique_lock<std::timed_mutex> lock(pool_mutexes_[i]);
    for (const auto &entry : pools_[i]) {
      if (entry.ptr == ptr && !entry.is_host_fallback) {
        return true;
      }
      if (entry.is_host_fallback && entry.host_ptr == ptr) {
        return false;
      }
    }
  }

  // If not found in pools, try cudaPointerGetAttributes to detect device ptr
  cudaPointerAttributes attr;
  cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
  if (err != cudaSuccess)
    return false;
#if CUDART_VERSION >= 10000
  return (attr.type == cudaMemoryTypeDevice);
#else
  // Older CUDA versions may use isManaged or other fields; fall back to
  // checking device
  return (attr.memoryType == cudaMemoryTypeDevice);
#endif
}

bool MemoryPoolManager::disable_host_fallback_env() {
  const char *env = getenv("CUDA_ZSTD_DISABLE_HOST_FALLBACK");
  if (!env)
    return false;
  return (std::string(env) == "1" || std::string(env) == "true");
}

bool MemoryPoolManager::auto_migrate_host_env() {
  const char *env = getenv("CUDA_ZSTD_AUTO_MIGRATE_HOST");
  if (!env)
    return false;
  return (std::string(env) == "1" || std::string(env) == "true");
}

void *MemoryPoolManager::migrate_host_to_device(void *host_ptr, size_t size,
                                                cudaStream_t stream) {
  if (!host_ptr || size == 0)
    return nullptr;

  // Allocate device memory
  void *d_ptr = nullptr;
  cudaError_t err = cudaMalloc(&d_ptr, size);
  if (err != cudaSuccess) {

    return nullptr;
  }

  // Copy host->device (async when possible)
  err = cudaMemcpyAsync(d_ptr, host_ptr, size, cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {

    cudaFree(d_ptr);
    return nullptr;
  }

  // If no stream provided, make sure copy finishes before returning
  if (stream == 0)
    cudaDeviceSynchronize();

  // Free host pointer

  free(host_ptr);

  // Log

  return d_ptr;
}

void *MemoryPoolManager::migrate_pool_host_entry_to_device(
    void *host_ptr, size_t size, cudaStream_t stream) {
  if (!host_ptr)
    return nullptr;

  // Search for the pool entry that has this host fallback pointer
  for (int i = 0; i < NUM_POOL_SIZES; ++i) {
    std::unique_lock<std::timed_mutex> lock(pool_mutexes_[i]);
    for (auto &entry : pools_[i]) {
      if (entry.is_host_fallback && entry.host_ptr == host_ptr) {
        // Attempt migration
        void *d_ptr = migrate_host_to_device(host_ptr, size, stream);
        if (!d_ptr)
          return nullptr;

        // Update pool entry metadata
        entry.ptr = d_ptr;
        entry.is_host_fallback = false;
        entry.host_ptr = nullptr;
        entry.host_size = 0;
        return d_ptr;
      }
    }
  }
  return nullptr;
}

// Note: migrate_host_to_device is implemented above; it is exposed in the
// header to allow external use when necessary.

FallbackAllocation
MemoryPoolManager::allocate_with_cuda_fallback(size_t size,
                                               cudaStream_t stream) {
  FallbackAllocation result;
  result.allocated_size = size;

  // Try primary CUDA allocation first
  result.ptr = allocate_from_cuda(size);
  if (result.ptr) {
    // CRITICAL: Check if this pointer already exists (CUDA can recycle
    // addresses)
    bool already_exists = false;
    {
      std::lock_guard<std::mutex> m(pointer_map_mutex_);
      if (pointer_index_map_.find(result.ptr) != pointer_index_map_.end()) {
        already_exists = true;
      }
    }

    // Also check if it exists in any pool
    if (!already_exists) {
      for (int i = 0; i < NUM_POOL_SIZES && !already_exists; ++i) {
        std::unique_lock<std::timed_mutex> lock(pool_mutexes_[i],
                                                std::defer_lock);
        if (lock.try_lock_for(std::chrono::milliseconds(10))) {
          for (const auto &entry : pools_[i]) {
            if (entry.ptr == result.ptr) {
              already_exists = true;
              break;
            }
          }
        }
      }
    }

    if (already_exists) {
      // Pointer already exists, free it and return error

      cudaFree(result.ptr);
      current_memory_usage_.fetch_sub(size, std::memory_order_relaxed);
      result.ptr = nullptr;
      result.status = Status::ERROR_OUT_OF_MEMORY;
      allocation_failures_.fetch_add(1, std::memory_order_relaxed);
      return result;
    }

    result.status = Status::SUCCESS;
    // Register pointer for tracking
    {
      std::lock_guard<std::mutex> m(pointer_map_mutex_);
      pointer_index_map_[result.ptr] = PointerMeta{
          -1, ++allocation_sequence_counter_, 0, nullptr, size, false};
    }
    return result;
  }

  // CUDA allocation failed, check if we should use fallback
  // We should fail ONLY if host memory fallback is disabled.
  // If is_emergency_mode() is true, we MUST allow proceeding to host fallback.
  if (!fallback_config_.enable_host_memory_fallback) {
    result.status = Status::ERROR_OUT_OF_MEMORY;
    allocation_failures_.fetch_add(1, std::memory_order_relaxed);
    return result;
  }

  // Try host memory fallback
  result.ptr = allocate_from_host(size);
  if (result.ptr) {
    result.is_host_memory = true;
    result.status = Status::SUCCESS;
    fallback_allocations_.fetch_add(1, std::memory_order_relaxed);

    // Register pointer for tracking (host memory)
    {
      std::lock_guard<std::mutex> m(pointer_map_mutex_);
      pointer_index_map_[result.ptr] = PointerMeta{
          -1, ++allocation_sequence_counter_, 0, nullptr, size, true};
    }

    // Respect environment toggle to disable host fallback
    if (MemoryPoolManager::disable_host_fallback_env()) {

      // Need to unregister before freeing
      {
        std::lock_guard<std::mutex> m(pointer_map_mutex_);
        pointer_index_map_.erase(result.ptr);
      }
      free(result.ptr);
      result.ptr = nullptr;
      result.status = Status::ERROR_OUT_OF_MEMORY;
      allocation_failures_.fetch_add(1, std::memory_order_relaxed);
      return result;
    }

    // Auto-migrate host->device if requested via env var
    if (MemoryPoolManager::auto_migrate_host_env()) {
      void *migrated = migrate_host_to_device(result.ptr, size, stream);
      if (migrated) {
        // Unregister old host pointer, register new device pointer
        {
          std::lock_guard<std::mutex> m(pointer_map_mutex_);
          pointer_index_map_.erase(result.ptr);
          pointer_index_map_[migrated] = PointerMeta{
              -1, ++allocation_sequence_counter_, 0, nullptr, size, false};
        }
        result.ptr = migrated;
        result.is_host_memory = false;
        result.is_dual_memory = true;
        result.allocation_path = "auto_migrated_from_host";

      } else {
      }
    }
    return result;
  }

  // Host allocation also failed
  result.status = Status::ERROR_OUT_OF_MEMORY;
  allocation_failures_.fetch_add(1, std::memory_order_relaxed);
  return result;
}

FallbackAllocation MemoryPoolManager::allocate_degraded(size_t size,
                                                        cudaStream_t stream) {
  FallbackAllocation result;

  // Calculate degraded size
  size_t degraded_size = calculate_degraded_size(size);
  if (degraded_size < 1024) { // Minimum 1KB
    degraded_size = 1024;
  }

  result.allocated_size = degraded_size;
  result.is_degraded = true;

  // Try degraded CUDA allocation
  result.ptr = allocate_from_cuda(degraded_size);
  if (result.ptr) {
    result.status = Status::SUCCESS;
    degraded_allocations_.fetch_add(1, std::memory_order_relaxed);
    // Register pointer for tracking
    {
      std::lock_guard<std::mutex> m(pointer_map_mutex_);
      pointer_index_map_[result.ptr] = PointerMeta{
          -1, ++allocation_sequence_counter_, 0, nullptr, degraded_size, false};
    }
    return result;
  }

  // Try degraded host allocation if fallback is enabled
  if (fallback_config_.enable_host_memory_fallback) {
    result.ptr = allocate_from_host(degraded_size);
    if (result.ptr) {
      result.is_host_memory = true;
      result.status = Status::SUCCESS;
      degraded_allocations_.fetch_add(1, std::memory_order_relaxed);
      fallback_allocations_.fetch_add(1, std::memory_order_relaxed);

      if (MemoryPoolManager::disable_host_fallback_env()) {

        free(result.ptr);
        result.ptr = nullptr;
        result.status = Status::ERROR_OUT_OF_MEMORY;
        allocation_failures_.fetch_add(1, std::memory_order_relaxed);
        return result;
      }

      if (MemoryPoolManager::auto_migrate_host_env()) {
        void *migrated =
            migrate_host_to_device(result.ptr, degraded_size, stream);
        if (migrated) {
          // Unregister old host pointer, register new device pointer
          {
            std::lock_guard<std::mutex> m(pointer_map_mutex_);
            pointer_index_map_.erase(result.ptr);
            pointer_index_map_[migrated] = PointerMeta{
                -1,   ++allocation_sequence_counter_, 0, nullptr, degraded_size,
                false};
          }
          result.ptr = migrated;
          result.is_host_memory = false;
          result.is_dual_memory = true;
          result.allocation_path = "auto_migrated_from_host_degraded";
        }
      }

      // Register pointer for tracking (if not migrated or migration failed,
      // it's still host memory)
      if (result.is_host_memory) {
        std::lock_guard<std::mutex> m(pointer_map_mutex_);
        pointer_index_map_[result.ptr] = PointerMeta{
            -1,  ++allocation_sequence_counter_, 0, nullptr, degraded_size,
            true};
      }

      return result;
    }
  }

  result.status = Status::ERROR_OUT_OF_MEMORY;
  allocation_failures_.fetch_add(1, std::memory_order_relaxed);
  return result;
}

Status MemoryPoolManager::grow_pool(int pool_idx, size_t min_entries) {
  if (pool_idx < 0 || pool_idx >= NUM_POOL_SIZES) {
    return Status::ERROR_INVALID_PARAMETER;
  }

  size_t entry_size = POOL_SIZES[pool_idx];
  size_t current_size = pools_[pool_idx].size();
  size_t new_entries =
      std::max(min_entries, static_cast<size_t>(current_size * growth_factor_));
  new_entries = std::max(new_entries, size_t(4)); // At least 4 new entries

  // Check if we would exceed max pool size
  size_t new_memory = new_entries * entry_size;
  size_t current_total = current_memory_usage_.load(std::memory_order_relaxed);
  if (current_total + new_memory > max_pool_size_) {
    // Try to allocate just what we need
    new_entries = min_entries;
    new_memory = new_entries * entry_size;
    if (current_total + new_memory > max_pool_size_) {
      // Pool growth would exceed max size, try fallback strategy
      return grow_pool_with_fallback(pool_idx, min_entries);
    }
  }

  // Allocate new entries with rollback protection
  std::vector<PoolEntry> new_pool_entries;
  new_pool_entries.reserve(new_entries);

  size_t successful_allocations = 0;
  size_t max_attempts =
      new_entries * 3; // Allow some duplicates without infinite loop
  size_t consecutive_duplicates = 0;

  for (size_t attempt = 0;
       attempt < max_attempts && successful_allocations < new_entries;
       ++attempt) {
    void *ptr = allocate_from_cuda(entry_size);
    if (!ptr) {
      // Allocation failed - if we have at least one successful allocation,
      // that's ok
      if (successful_allocations > 0) {
        break;
      }
      // Otherwise, this is a failure
      trigger_rollback_protection();
      return Status::ERROR_OUT_OF_MEMORY;
    }

    // Check if this pointer already exists in ANY pool (CUDA can recycle freed
    // addresses) We must check ALL pool indices, not just the current one, to
    // prevent the same pointer from being added to multiple pools
    bool duplicate = false;

    // Check all existing pools
    // Check all existing pools
    for (int check_idx = 0; check_idx < NUM_POOL_SIZES && !duplicate;
         ++check_idx) {
      // Lock the pool we're checking to avoid race conditions
      std::unique_lock<std::timed_mutex> check_lock(pool_mutexes_[check_idx],
                                                    std::defer_lock);

      // Always try to lock, even for pool_idx, because caller
      // (allocate/prewarm) does NOT hold the lock when calling grow_pool.
      if (check_lock.try_lock_for(std::chrono::milliseconds(100))) {
        for (const auto &existing_entry : pools_[check_idx]) {
          if (existing_entry.ptr == ptr) {
            duplicate = true;
            break;
          }
        }
      } else {
        // If we can't lock, we skip checking this pool.
        // This is a trade-off: we might miss a duplicate, but we avoid
        // deadlock. Given duplicates are rare (only if CUDA recycles), this is
        // acceptable.
      }
    }

    // Also check against newly allocated entries in this batch
    if (!duplicate) {
      for (const auto &new_entry : new_pool_entries) {
        if (new_entry.ptr == ptr) {
          duplicate = true;
          break;
        }
      }
    }

    if (duplicate) {
      // Free the duplicate immediately to avoid memory leaks
      cudaFree(ptr);
      current_memory_usage_.fetch_sub(entry_size, std::memory_order_relaxed);
      consecutive_duplicates++;

      // If we get too many consecutive duplicates, CUDA is stuck recycling the
      // same addresses Break early to avoid infinite loop
      if (consecutive_duplicates > 10 && successful_allocations > 0) {

        break;
      }
      continue;
    }

    consecutive_duplicates = 0; // Reset counter on successful allocation
    new_pool_entries.emplace_back(ptr, entry_size);
    // Assign UID so that pointer map entries can detect reused slots
    new_pool_entries.back().uid = ++entry_uid_counter_;
    successful_allocations++;
  }

  if (successful_allocations == 0) {
    return Status::ERROR_OUT_OF_MEMORY;
  }

  // All allocations succeeded, add to pool
  // Lock the requested pool using a timed mutex so we can recover from
  // pathological blocking conditions. The default behaviour (when the
  // env var is not set) is to block indefinitely and preserve semantics.
  const char *env_lock = getenv("CUDA_ZSTD_POOL_LOCK_TIMEOUT_MS");
  unsigned long timeout_ms = 0;
  if (env_lock) {
    try {
      timeout_ms = std::stoul(env_lock);
    } catch (...) {
      timeout_ms = 0;
    }
  }

  std::unique_lock<std::timed_mutex> lock(pool_mutexes_[pool_idx],
                                          std::defer_lock);
  if (timeout_ms == 0) {
    lock.lock();
  } else {
    if (!lock.try_lock_for(std::chrono::milliseconds(timeout_ms))) {

      return Status::ERROR_TIMEOUT;
    }
  }
  pools_[pool_idx].insert(pools_[pool_idx].end(),
                          std::make_move_iterator(new_pool_entries.begin()),
                          std::make_move_iterator(new_pool_entries.end()));

  pool_grows_.fetch_add(1, std::memory_order_relaxed);
  return Status::SUCCESS;
}

Status MemoryPoolManager::grow_pool_with_fallback(int pool_idx,
                                                  size_t min_entries) {
  if (!fallback_config_.enable_progressive_degradation) {
    return Status::ERROR_OUT_OF_MEMORY;
  }

  // Try with reduced entry size
  size_t original_entry_size = POOL_SIZES[pool_idx];
  size_t degraded_entry_size = calculate_degraded_size(original_entry_size);

  if (degraded_entry_size < SIZE_4KB) {
    degraded_entry_size = SIZE_4KB; // Minimum pool size
  }

  // Try to grow with smaller entries
  for (size_t attempt = 0; attempt < fallback_config_.max_retry_attempts;
       ++attempt) {
    void *ptr = allocate_from_cuda(degraded_entry_size);
    if (ptr) {
      std::unique_lock<std::timed_mutex> lock(pool_mutexes_[pool_idx]);
      pools_[pool_idx].emplace_back(ptr, degraded_entry_size);
      degraded_allocations_.fetch_add(1, std::memory_order_relaxed);
      pool_grows_.fetch_add(1, std::memory_order_relaxed);
      return Status::SUCCESS;
    }

    // If this is the last attempt, try host memory fallback
    if (attempt == fallback_config_.max_retry_attempts - 1 &&
        fallback_config_.enable_host_memory_fallback) {
      ptr = allocate_from_host(degraded_entry_size);
      if (ptr) {

        if (MemoryPoolManager::disable_host_fallback_env()) {

          free(ptr);
          continue;
        }
        std::unique_lock<std::timed_mutex> lock(pool_mutexes_[pool_idx]);
        PoolEntry entry;
        entry.ptr = nullptr; // No GPU memory
        entry.host_ptr = ptr;
        entry.size = 0;
        entry.host_size = degraded_entry_size;
        entry.is_host_fallback = true;
        entry.in_use = false;
        pools_[pool_idx].push_back(std::move(entry));

        host_memory_allocations_.fetch_add(1, std::memory_order_relaxed);
        fallback_allocations_.fetch_add(1, std::memory_order_relaxed);
        pool_grows_.fetch_add(1, std::memory_order_relaxed);
        return Status::SUCCESS;
      }
    }
  }

  return Status::ERROR_OUT_OF_MEMORY;
}

PoolEntry *MemoryPoolManager::find_free_entry(int pool_idx,
                                              cudaStream_t stream) {
  auto &pool = pools_[pool_idx];

  for (auto &entry : pool) {
    if (!entry.in_use) {
      // If this entry is a host-fallback we may either skip it
      // (if disabled) or attempt to migrate it to device memory
      // on demand (if auto-migrate env is set). This avoids
      // returning host pointers into kernel contexts.
      if (entry.is_host_fallback) {
        if (MemoryPoolManager::disable_host_fallback_env()) {
          continue; // Treat as unavailable
        }

        // Attempt on-demand migration for test runs
        if (MemoryPoolManager::auto_migrate_host_env()) {
          void *migrated =
              migrate_host_to_device(entry.host_ptr, entry.host_size, stream);
          if (migrated) {

            entry.ptr = migrated;
            entry.host_ptr = nullptr;
            entry.size = entry.host_size;
            entry.host_size = 0;
            entry.is_host_fallback = false;
            entry.is_degraded = false;
            // Count this as a fallback->device migration
            fallback_allocations_.fetch_add(1, std::memory_order_relaxed);
          } else {
            // If migration fails, avoid using the host fallback pointer
            continue;
          }
        } else {
          // Not auto-migrating — skip this entry for GPU allocations
          continue;
        }
      }
      // Check if the entry is ready (stream synchronization)
      if (entry.ready_event != nullptr) {
        cudaError_t err = cudaEventQuery(entry.ready_event);
        if (err == cudaErrorNotReady) {
          continue; // Entry not ready yet
        } else if (err == cudaSuccess) {
          cudaEventDestroy(entry.ready_event);
          entry.ready_event = nullptr;
        } else {
          // Error querying event, destroy it anyway
          cudaEventDestroy(entry.ready_event);
          entry.ready_event = nullptr;
        }
      }

      // CRITICAL: Check if this pointer EXISTS in ANY other pool
      // (duplicate pointers can exist across pool indices from before we added
      // duplicate prevention) We must check for existence, not just in-use
      // status, because after deallocation the pointer in one pool becomes
      // free, but if it exists in another pool too, we'll return it again
      // causing double-allocation
      bool exists_elsewhere = false;
      for (int check_idx = 0; check_idx < NUM_POOL_SIZES && !exists_elsewhere;
           ++check_idx) {
        if (check_idx == pool_idx)
          continue; // Skip current pool

        // Try to lock other pool briefly to check
        std::unique_lock<std::timed_mutex> check_lock(pool_mutexes_[check_idx],
                                                      std::defer_lock);
        if (check_lock.try_lock_for(std::chrono::milliseconds(10))) {
          for (const auto &other_entry : pools_[check_idx]) {
            // Check if pointer exists at all, regardless of in-use status
            if (other_entry.ptr == entry.ptr) {
              exists_elsewhere = true;
              break;
            }
          }
        }
      }

      if (exists_elsewhere) {
        // This pointer exists in another pool, skip this entry entirely
        // to prevent double-allocation
        continue;
      }

      entry.in_use = true;
      entry.stream = stream;

      return &entry;
    }
  }

  return nullptr;
}

// ============================================================================
// Allocation Interface with Fallback Support
// ============================================================================

void *MemoryPoolManager::allocate(size_t size, cudaStream_t stream) {
  if (size == 0) {
    return nullptr;
  }

  total_allocations_.fetch_add(1, std::memory_order_relaxed);

  // Update degradation mode based on memory pressure
  update_degradation_mode();

  int pool_idx = get_pool_index(size);

  // For very large allocations, bypass the pool and use fallback strategy
  if (pool_idx < 0) {
    cache_misses_.fetch_add(1, std::memory_order_relaxed);

    // Use fallback allocation for large allocations
    FallbackAllocation result = allocate_with_cuda_fallback(size, stream);
    if (result.is_host_memory) {
      // For host memory allocations, caller must handle data movement
      return result.ptr;
    }
    return result.ptr;
  }

  unsigned pool_lock_timeout = get_pool_lock_timeout_ms();
  std::unique_lock<std::timed_mutex> lock(pool_mutexes_[pool_idx],
                                          std::defer_lock);
  if (pool_lock_timeout == 0) {
    lock.lock();
  } else {
    if (!lock.try_lock_for(std::chrono::milliseconds(pool_lock_timeout))) {
      //             std::cerr << "MemoryPoolManager::allocate: timeout
      //             acquiring lock for pool_idx=" << pool_idx << " after " <<
      //             pool_lock_timeout << "ms" << std::endl;
      allocation_failures_.fetch_add(1, std::memory_order_relaxed);
      return nullptr;
    }
  }
  //     // std::cerr << "MemoryPoolManager::allocate: got lock for pool_idx="
  //     << pool_idx << std::endl;

  // Try to find a free entry in the pool
  PoolEntry *entry = find_free_entry(pool_idx, stream);
  //     // std::cerr << "MemoryPoolManager::allocate: after find_free_entry
  //     entry=" << entry << std::endl;

  if (entry) {
    cache_hits_.fetch_add(1, std::memory_order_relaxed);
    entry->in_use = true;
    // Record allocation sequence and map pointer to pool index for
    // deterministic deallocation and race-detection.
    entry->alloc_seq = ++allocation_sequence_counter_;
    {
      std::lock_guard<std::mutex> m(pointer_map_mutex_);
      pointer_index_map_[entry->ptr] = PointerMeta{
          pool_idx, entry->alloc_seq, entry->uid, nullptr, 0, false};
    }
    //         std::cerr << "allocate: returning pool ptr=" << entry->ptr <<
    //         "\n";
    return entry->ptr;
  }

  // No free entry found, try to grow the pool
  cache_misses_.fetch_add(1, std::memory_order_relaxed);
  // Release lock before attempting to grow the pool to avoid deadlock
  lock.unlock();
  Status status = grow_pool(pool_idx, 1);
  // Reacquire lock after growth attempt
  lock.lock();
  //     // std::cerr << "MemoryPoolManager::allocate: grow_pool returned
  //     status=" << (int)status << std::endl;
  if (status == Status::SUCCESS) {
    // Try again after growing
    entry = find_free_entry(pool_idx, stream);
    if (entry) {
      entry->alloc_seq = ++allocation_sequence_counter_;
      {
        std::lock_guard<std::mutex> m(pointer_map_mutex_);
        pointer_index_map_[entry->ptr] = PointerMeta{
            pool_idx, entry->alloc_seq, entry->uid, nullptr, 0, false};
      }
      //             std::cerr << "allocate: returning grown pool ptr=" <<
      //             entry->ptr << "\n";
      return entry->ptr;
    }
  }

  // Pool growth failed, try fallback allocation
  FallbackAllocation result = allocate_with_cuda_fallback(size, stream);
  if (result.is_valid()) {
    //         std::cerr << "allocate: returning fallback ptr=" << result.ptr <<
    //         "\n";
    return result.ptr;
  }

  return nullptr;
}

FallbackAllocation
MemoryPoolManager::allocate_with_fallback(size_t requested_size,
                                          cudaStream_t stream) {
  FallbackAllocation result;
  result.allocated_size = requested_size;

  // Update degradation mode
  update_degradation_mode();

  // Try normal allocation first
  void *ptr = allocate(requested_size, stream);
  if (ptr) {
    result.ptr = ptr;
    result.status = Status::SUCCESS;
    // Fix: allocate() returns void* but might return host memory in fallback
    // cases. We need to check if the returned pointer is device memory to set
    // the flag correctly.
    result.is_host_memory = !is_device_pointer(ptr);
    return result;
  }

  // Normal allocation failed, try progressive allocation if enabled
  if (fallback_config_.enable_progressive_degradation) {
    result = allocate_progressive(requested_size / 2, requested_size, stream);
    if (result.is_valid()) {
      return result;
    }
  }

  // All strategies failed
  result.status = Status::ERROR_OUT_OF_MEMORY;
  allocation_failures_.fetch_add(1, std::memory_order_relaxed);
  return result;
}

FallbackAllocation
MemoryPoolManager::allocate_progressive(size_t min_size, size_t max_size,
                                        cudaStream_t stream) {
  FallbackAllocation result;

  // Progressive allocation: start from max and reduce until successful
  std::vector<size_t> try_sizes;
  size_t current_size = max_size;

  while (current_size >= min_size && current_size >= SIZE_4KB) {
    try_sizes.push_back(current_size);
    current_size =
        static_cast<size_t>(current_size * fallback_config_.degradation_factor);
    if (current_size < SIZE_4KB)
      break;
  }

  // Try each size
  for (size_t try_size : try_sizes) {
    result = allocate_with_cuda_fallback(try_size, stream);
    if (result.is_valid()) {
      result.is_degraded = (try_size < max_size);
      if (result.is_degraded) {
        degraded_allocations_.fetch_add(1, std::memory_order_relaxed);
      }
      return result;
    }
  }

  result.status = Status::ERROR_OUT_OF_MEMORY;
  return result;
}

void *MemoryPoolManager::allocate_async(size_t size, cudaStream_t stream) {
  void *ptr = allocate(size, stream);

  // For async allocations, we don't wait for the stream
  // The caller is responsible for stream synchronization
  return ptr;
}

Status MemoryPoolManager::deallocate(void *ptr) {
  if (!ptr) {
    return Status::SUCCESS;
  }

  total_deallocations_.fetch_add(1, std::memory_order_relaxed);

  // Search all pools to find this pointer
  bool timeout_occurred = false;
  for (int i = 0; i < NUM_POOL_SIZES; ++i) {
    // guard is a timed mutex that by default blocks (timeout=0). For test
    // environments we optionally use a try_lock timeout.
    const char *env_lock_all = getenv("CUDA_ZSTD_POOL_LOCK_TIMEOUT_MS");
    unsigned long lock_timeout_all =
        100; // Default to 100ms to prevent deadlocks
    if (env_lock_all) {
      try {
        lock_timeout_all = std::stoul(env_lock_all);
      } catch (...) {
        lock_timeout_all = 100;
      }
    }

    std::unique_lock<std::timed_mutex> lock(pool_mutexes_[i], std::defer_lock);
    if (!lock.try_lock_for(std::chrono::milliseconds(lock_timeout_all))) {
      // If we can't acquire the lock, we can't safely check this pool.
      // We must track that we missed a pool check due to timeout.
      timeout_occurred = true;
      continue;
    }

    for (auto &entry : pools_[i]) {
      if (entry.ptr == ptr) {
        if (!entry.in_use) {
          //                     std::cerr << "MemoryPoolManager::deallocate:
          //                     DOUBLE FREE detected in pool_idx=" << i
          //                               << " ptr=" << ptr << " entry.in_use="
          //                               << entry.in_use << "\n";
          return Status::ERROR_INVALID_PARAMETER; // Double free
        }

        // Handle stream synchronization
        if (entry.stream != nullptr && entry.stream != 0) {
          cudaEvent_t event;
          cudaError_t err = cudaEventCreate(&event);
          if (err == cudaSuccess) {
            cudaEventRecord(event, entry.stream);
            entry.ready_event = event;
          }
        }

        entry.in_use = false;
        entry.stream = nullptr;
        return Status::SUCCESS;
      }

      // Check for host memory fallback pointers
      if (entry.is_host_fallback && entry.host_ptr == ptr) {
        if (!entry.in_use) {
          return Status::ERROR_INVALID_PARAMETER; // Double free
        }

        //                 std::cerr << "MemoryPoolManager::deallocate: freeing
        //                 host fallback ptr=" << entry.host_ptr
        //                           << " size=" << entry.host_size << "
        //                           pool_idx=" << i << "\n";
        // Use debug_free wrapper to capture a stacktrace and free safely
        cuda_zstd::util::debug_free(entry.host_ptr);
        entry.host_ptr = nullptr;
        entry.in_use = false;

        // Update host memory usage
        host_memory_usage_.fetch_sub(entry.host_size,
                                     std::memory_order_relaxed);

        return Status::SUCCESS;
      }
    }
  }

  // Not in pool, must be a direct allocation
  // Check if it's host memory (we can't easily distinguish)
  // For safety, try both CUDA and host deallocation

  // Check if it's a tracked direct allocation (from allocate_with_strategy,
  // etc.)
  {
    std::lock_guard<std::mutex> m(pointer_map_mutex_);
    auto it = pointer_index_map_.find(ptr);
    if (it != pointer_index_map_.end()) {
      // Found in tracking map - this is a direct allocation (pool_idx=-1)
      int pool_idx = it->second.pool_idx;
      bool is_host = it->second.is_host_memory;
      size_t alloc_size = it->second.size;
      // Note: host_ptr is caller-managed, we don't free it here
      pointer_index_map_.erase(it);

      if (pool_idx == -1) {
        // Direct allocation - free with correct function
        if (is_host) {
          // Host memory - use free()
          cuda_zstd::util::debug_free(ptr);
          host_memory_usage_.fetch_sub(alloc_size, std::memory_order_relaxed);
        } else {
          // GPU memory - use cudaFree()
          cudaError_t err = cudaFree(ptr);
          if (err != cudaSuccess) {
            //                         std::cerr <<
            //                         "MemoryPoolManager::deallocate: cudaFree
            //                         failed for tracked ptr=" << ptr
            //                                   << " err=" << err << std::endl;
            return Status::ERROR_CUDA_ERROR;
          }
        }
        return Status::SUCCESS;
      }
    }
  }

  //     std::cerr << "MemoryPoolManager::deallocate: pointer not found in pools
  //     or tracking map. ptr=" << ptr << std::endl;
  if (timeout_occurred) {
    return Status::ERROR_TIMEOUT;
  }
  return Status::ERROR_INVALID_PARAMETER;

  // Not in pool, must be a direct allocation. For portability and safety
  // avoid calling cudaFree blindly in debug/memcheck runs where pointers
  // might have been corrupted by kernel bugs. Instead, attempt a safe
  // CUDA deallocation only if this appears to be a pointer returned by
  // our pool fallback. We do not have a perfect way to distinguish
  // allocated pointers; to avoid double-crash we log and return error.
  // Nothing matched; do not call cudaFree for unknown pointer. This
  // prevents invalid argument errors when kernels corrupted pointers
  // or when the pointer belongs to a caller-managed buffer.
  //     std::cerr << "MemoryPoolManager::deallocate: pointer not found in
  //     pools: ptr=" << ptr << " - skipping free" << std::endl;
  return Status::ERROR_INVALID_PARAMETER;
}

// ============================================================================
// Pool Management
// ============================================================================

Status MemoryPoolManager::prewarm(size_t total_memory) {
  // Distribute memory across pool sizes proportionally
  // Strategy: Smaller sizes get more entries, larger sizes get fewer

  // remaining was previously unused; memory distribution is computed per-pool

  // Weights for each pool size (smaller = more weight)
  const float weights[NUM_POOL_SIZES] = {4.0f, 3.0f, 2.5f, 2.0f, 1.5f, 1.0f};
  float total_weight = 0.0f;
  for (float w : weights)
    total_weight += w;

  for (int i = 0; i < NUM_POOL_SIZES; ++i) {
    size_t pool_memory =
        static_cast<size_t>(total_memory * weights[i] / total_weight);
    size_t num_entries = pool_memory / POOL_SIZES[i];

    if (num_entries > 0) {
      // Do NOT lock here. grow_pool handles locking internally.
      // Locking here would cause a deadlock because grow_pool tries to acquire
      // the same non-recursive mutex.
      // std::unique_lock<std::timed_mutex> lock(pool_mutexes_[i]);
      Status status = grow_pool(i, num_entries);
      if (status != Status::SUCCESS) {
        return status;
      }
    }
  }

  return Status::SUCCESS;
}

Status MemoryPoolManager::defragment() {
  if (!defrag_enabled_) {
    return Status::SUCCESS;
  }

  // Defragmentation strategy:
  // 1. Identify pools with many free entries
  // 2. Free entries at the end of the pool vector
  // 3. Compact the pool by removing freed entries

  for (int i = 0; i < NUM_POOL_SIZES; ++i) {
    std::unique_lock<std::timed_mutex> lock(pool_mutexes_[i]);
    auto &pool = pools_[i];

    // Count free entries
    size_t free_count = 0;
    for (const auto &entry : pool) {
      if (!entry.in_use && entry.ready_event == nullptr) {
        free_count++;
      }
    }

    // If more than 50% are free, compact
    if (free_count > pool.size() / 2 && pool.size() > 8) {
      std::vector<PoolEntry> compacted;
      compacted.reserve(pool.size() - free_count / 2);

      size_t freed_memory = 0;

      for (auto &entry : pool) {
        if (entry.in_use || entry.ready_event != nullptr) {
          compacted.push_back(std::move(entry));
        } else if (compacted.size() < pool.size() - free_count / 2) {
          compacted.push_back(std::move(entry));
        } else {
          // Free this entry
          if (entry.ptr) {
            cudaFree(entry.ptr);
            freed_memory += entry.size;
          }
        }
      }

      pool = std::move(compacted);
      current_memory_usage_.fetch_sub(freed_memory, std::memory_order_relaxed);
    }
  }

  return Status::SUCCESS;
}

void MemoryPoolManager::clear() {
  // Lock all pools in deterministic order before clearing to avoid
  // deadlocks when another thread concurrently tries to lock multiple
  // pools. `lock_pools_ordered` will honor the configured timeout.
  std::vector<int> all_indices;
  all_indices.reserve(NUM_POOL_SIZES);
  for (int i = 0; i < NUM_POOL_SIZES; ++i)
    all_indices.push_back(i);

  std::vector<std::unique_lock<std::timed_mutex>> locks;
  // By default block indefinitely (preserve original semantics)
  lock_pools_ordered(all_indices, 0, locks);

  for (int i = 0; i < NUM_POOL_SIZES; ++i) {
    for (auto &entry : pools_[i]) {
      if (entry.ready_event != nullptr) {
        cudaEventSynchronize(entry.ready_event);
        cudaEventDestroy(entry.ready_event);
        entry.ready_event = nullptr;
      }
      if (entry.ptr && !entry.is_host_fallback) {
        cudaFree(entry.ptr);
        entry.ptr = nullptr;
      }
      if (entry.host_ptr) {
        free(entry.host_ptr);
        entry.host_ptr = nullptr;
      }
    }

    pools_[i].clear();
  }
  // std::cerr << "MemoryPoolManager::clear() finished\n";

  current_memory_usage_.store(0, std::memory_order_relaxed);
  host_memory_usage_.store(0, std::memory_order_relaxed);
}

void MemoryPoolManager::reset_for_reuse() {
  // Reset pool state between compressions WITHOUT destroying pool memory
  // This keeps pool performance while ensuring clean state

  // Lock all pools to ensure thread safety
  std::vector<int> all_indices;
  all_indices.reserve(NUM_POOL_SIZES);
  for (int i = 0; i < NUM_POOL_SIZES; ++i)
    all_indices.push_back(i);

  std::vector<std::unique_lock<std::timed_mutex>> locks;
  bool locked = lock_pools_ordered(all_indices, 500, locks); // 500ms timeout

  // If we couldn't acquire locks, skip reset to avoid deadlock
  if (!locked || locks.empty()) {
    return;
  }

  // 1. Destroy any pending events with proper error handling
  for (int i = 0; i < NUM_POOL_SIZES; ++i) {
    for (auto &entry : pools_[i]) {
      if (entry.ready_event != nullptr) {
        // Synchronize with error handling - ignore errors on stale events
        cudaError_t sync_err = cudaEventSynchronize(entry.ready_event);
        if (sync_err != cudaSuccess) {
          // Event is likely corrupted or already destroyed, just reset pointer
          cudaGetLastError(); // Clear error state
        }

        cudaError_t destroy_err = cudaEventDestroy(entry.ready_event);
        if (destroy_err != cudaSuccess) {
          cudaGetLastError(); // Clear error state
        }
        entry.ready_event = nullptr;
      }
      // Clear stream association
      entry.stream = nullptr;
      // Force clear in_use flag to handle leaks/stale state
      entry.in_use = false;
    }
  }

  // 2. Reset error counters (but keep cache statistics)
  allocation_failures_.store(0, std::memory_order_relaxed);
  rollback_operations_.store(0, std::memory_order_relaxed);

  // 3. Reset degradation mode to NORMAL
  {
    std::lock_guard<std::timed_mutex> mode_lock(mode_mutex_);
    current_mode_ = DegradationMode::NORMAL;
  }

  // 4. Clear any stale error states
  last_pressure_update_ = std::chrono::steady_clock::now();

  // Note: We do NOT clear the pools themselves or free memory
  // The pool stays alive for performance
}

Status MemoryPoolManager::emergency_clear() {
  std::lock_guard<std::timed_mutex> mode_lock(mode_mutex_);

  // Switch to emergency mode
  current_mode_ = DegradationMode::EMERGENCY;

  // Clear all pools
  clear();

  // Reset statistics
  reset_statistics();

  return Status::SUCCESS;
}

Status MemoryPoolManager::switch_to_host_memory_mode() {
  std::lock_guard<std::timed_mutex> mode_lock(mode_mutex_);

  if (!fallback_config_.enable_host_memory_fallback) {
    return Status::ERROR_INVALID_PARAMETER;
  }

  // Switch to emergency mode and clear GPU pools
  current_mode_ = DegradationMode::EMERGENCY;
  clear();

  return Status::SUCCESS;
}

// ============================================================================
// Statistics
// ============================================================================

PoolStats MemoryPoolManager::get_statistics() const {
  PoolStats stats;
  stats.total_allocations = total_allocations_.load(std::memory_order_relaxed);
  stats.total_deallocations =
      total_deallocations_.load(std::memory_order_relaxed);
  stats.cache_hits = cache_hits_.load(std::memory_order_relaxed);
  stats.cache_misses = cache_misses_.load(std::memory_order_relaxed);
  stats.pool_grows = pool_grows_.load(std::memory_order_relaxed);
  stats.fallback_allocations =
      fallback_allocations_.load(std::memory_order_relaxed);
  stats.host_memory_allocations =
      host_memory_allocations_.load(std::memory_order_relaxed);
  stats.degraded_allocations =
      degraded_allocations_.load(std::memory_order_relaxed);
  stats.allocation_failures =
      allocation_failures_.load(std::memory_order_relaxed);
  stats.rollback_operations =
      rollback_operations_.load(std::memory_order_relaxed);
  stats.peak_memory_usage = peak_memory_usage_.load(std::memory_order_relaxed);
  stats.current_memory_usage =
      current_memory_usage_.load(std::memory_order_relaxed);
  stats.host_memory_usage = host_memory_usage_.load(std::memory_order_relaxed);
  stats.current_mode = current_mode_;

  // Calculate total pool capacity
  size_t total_capacity = 0;
  for (int i = 0; i < NUM_POOL_SIZES; ++i) {
    std::unique_lock<std::timed_mutex> lock(pool_mutexes_[i]);
    total_capacity += pools_[i].size() * POOL_SIZES[i];
  }
  stats.total_pool_capacity = total_capacity;

  return stats;
}

void MemoryPoolManager::reset_statistics() {
  total_allocations_.store(0, std::memory_order_relaxed);
  total_deallocations_.store(0, std::memory_order_relaxed);
  cache_hits_.store(0, std::memory_order_relaxed);
  cache_misses_.store(0, std::memory_order_relaxed);
  pool_grows_.store(0, std::memory_order_relaxed);
  fallback_allocations_.store(0, std::memory_order_relaxed);
  host_memory_allocations_.store(0, std::memory_order_relaxed);
  degraded_allocations_.store(0, std::memory_order_relaxed);
  allocation_failures_.store(0, std::memory_order_relaxed);
  rollback_operations_.store(0, std::memory_order_relaxed);
  // Note: We don't reset peak_memory_usage as it's cumulative
}

void MemoryPoolManager::print_statistics() const {
  // No-op: statistics printing not implemented
  // Use get_statistics() to retrieve pool stats programmatically.
  (void)get_statistics();
}

// ============================================================================
// Configuration
// ============================================================================

void MemoryPoolManager::set_growth_factor(float factor) {
  if (factor >= 1.0f) {
    growth_factor_ = factor;
  }
}

void MemoryPoolManager::enable_defragmentation(bool enable) {
  defrag_enabled_ = enable;
}

void MemoryPoolManager::set_max_pool_size(size_t max_size) {
  max_pool_size_ = max_size;
}

// ============================================================================
// Fallback Configuration Methods
// ============================================================================

void MemoryPoolManager::set_fallback_config(const FallbackConfig &config) {
  std::lock_guard<std::timed_mutex> lock(mode_mutex_);
  fallback_config_ = config;
}

const FallbackConfig &MemoryPoolManager::get_fallback_config() const {
  std::lock_guard<std::timed_mutex> lock(mode_mutex_);
  return fallback_config_;
}

void MemoryPoolManager::set_degradation_mode(DegradationMode mode) {
  std::lock_guard<std::timed_mutex> lock(mode_mutex_);
  current_mode_ = mode;
}

DegradationMode MemoryPoolManager::get_degradation_mode() const {
  std::lock_guard<std::timed_mutex> lock(mode_mutex_);
  return current_mode_;
}

// ============================================================================
// Memory Pressure Monitoring
// ============================================================================

size_t MemoryPoolManager::get_available_gpu_memory() const {
  return get_available_gpu_memory_impl();
}

size_t MemoryPoolManager::get_available_gpu_memory_impl() const {
  size_t free_mem = 0;
  size_t total_mem = 0;

  cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
  if (err == cudaSuccess) {
    return free_mem;
  }

  return 0; // Return 0 if we can't get memory info
}

size_t MemoryPoolManager::get_memory_pressure_percentage() const {
  size_t total_mem = 0;
  size_t free_mem = get_available_gpu_memory_impl();

  cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
  if (err != cudaSuccess || total_mem == 0) {
    return 0;
  }

  size_t used_mem = total_mem - free_mem;
  return static_cast<size_t>((static_cast<double>(used_mem) / total_mem) *
                             100.0);
}

bool MemoryPoolManager::is_memory_pressure_high() const {
  return is_memory_pressure_critical();
}

bool MemoryPoolManager::is_memory_pressure_critical() const {
  auto now = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::seconds>(
                      now - last_pressure_update_)
                      .count();

  // Only check every few seconds to avoid overhead
  if (duration < 3) {
    return current_mode_ == DegradationMode::EMERGENCY;
  }

  size_t pressure = get_memory_pressure_percentage();
  last_pressure_update_ = now;

  // Switch to emergency mode if pressure is very high
  if (pressure > 90) {
    return true;
  }

  return false;
}

void MemoryPoolManager::update_degradation_mode() {
  std::lock_guard<std::timed_mutex> lock(mode_mutex_);

  // Only update if not already in emergency mode
  if (current_mode_ == DegradationMode::EMERGENCY) {
    return;
  }

  size_t pressure = get_memory_pressure_percentage();

  if (pressure > 90) {
    current_mode_ = DegradationMode::EMERGENCY;
  } else if (pressure > 75) {
    current_mode_ = DegradationMode::AGGRESSIVE;
  } else if (pressure > 60) {
    current_mode_ = DegradationMode::CONSERVATIVE;
  } else {
    current_mode_ = DegradationMode::NORMAL;
  }
}

// ============================================================================
// Helper Methods for Fallback Logic
// ============================================================================

size_t MemoryPoolManager::calculate_degraded_size(size_t original_size) const {
  std::lock_guard<std::timed_mutex> lock(mode_mutex_);

  switch (current_mode_) {
  case DegradationMode::NORMAL:
    return original_size;
  case DegradationMode::CONSERVATIVE:
    return static_cast<size_t>(original_size * 0.8f); // 20% reduction
  case DegradationMode::AGGRESSIVE:
    return static_cast<size_t>(original_size * 0.5f); // 50% reduction
  case DegradationMode::EMERGENCY:
    return static_cast<size_t>(original_size * 0.25f); // 75% reduction
  default:
    return original_size;
  }
}

bool MemoryPoolManager::should_use_progressive_allocation(
    size_t requested_size) const {
  if (!fallback_config_.enable_progressive_degradation) {
    return false;
  }

  // Use progressive allocation for large requests
  return requested_size > SIZE_1MB;
}

bool MemoryPoolManager::is_emergency_mode() const {
  std::lock_guard<std::timed_mutex> lock(mode_mutex_);
  return current_mode_ == DegradationMode::EMERGENCY;
}

void MemoryPoolManager::trigger_rollback_protection() {
  rollback_operations_.fetch_add(1, std::memory_order_relaxed);

  // In case of repeated failures, switch to more conservative mode
  if (allocation_failures_.load(std::memory_order_relaxed) > 10) {
    std::lock_guard<std::timed_mutex> lock(mode_mutex_);
    if (current_mode_ == DegradationMode::NORMAL) {
      current_mode_ = DegradationMode::CONSERVATIVE;
    } else if (current_mode_ == DegradationMode::CONSERVATIVE) {
      current_mode_ = DegradationMode::AGGRESSIVE;
    }
  }
}

Status MemoryPoolManager::copy_between_memory_types(void *src, void *dst,
                                                    size_t size,
                                                    bool host_to_device) {
  cudaMemcpyKind kind =
      host_to_device ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
  cudaError_t err = cudaMemcpy(dst, src, size, kind);
  return (err == cudaSuccess) ? Status::SUCCESS : Status::ERROR_CUDA_ERROR;
}

// ============================================================================
// Smart Allocation Interface
// ============================================================================

FallbackAllocation
MemoryPoolManager::allocate_smart(const AllocationContext &context) {
  // Log the request
  //     // std::cout << "Smart allocation request: " << context.requested_size
  //     << " bytes, strategy=" << (int)context.strategy << "\n";

  // Update resource state if needed
  if (std::chrono::steady_clock::now() - last_resource_update_ >
      std::chrono::milliseconds(100)) {
    update_resource_state();
  }

  AllocationStrategy strategy = context.strategy;
  if (strategy == AllocationStrategy::AUTO_ADAPTIVE) {
    strategy =
        select_optimal_strategy(cached_resource_state_, context.requested_size);
  }

  return allocate_with_strategy(context.requested_size, strategy,
                                context.stream);
}

FallbackAllocation MemoryPoolManager::allocate_with_strategy(
    size_t size, AllocationStrategy strategy, cudaStream_t stream) {
  FallbackAllocation result;
  result.requested_size = size;
  result.strategy_used = strategy;

  auto start_time = std::chrono::high_resolution_clock::now();

  switch (strategy) {
  case AllocationStrategy::PREFER_GPU:
    result = allocate_with_cuda_fallback(size, stream);
    break;

  case AllocationStrategy::PREFER_HOST:
    result.ptr = allocate_from_host(size);
    if (result.ptr) {
      result.status = Status::SUCCESS;
      result.is_host_memory = true;
      result.allocated_size = size;
    } else {
      result.status = Status::ERROR_OUT_OF_MEMORY;
    }
    break;

  case AllocationStrategy::BALANCED:
    // Try GPU first, but be quick to fallback if pressure is high
    if (is_memory_pressure_high()) {
      result.ptr = allocate_from_host(size);
      if (result.ptr) {
        result.status = Status::SUCCESS;
        result.is_host_memory = true;
        result.allocated_size = size;
      } else {
        // If host fails, try GPU as last resort
        result = allocate_with_cuda_fallback(size, stream);
      }
    } else {
      result = allocate_with_cuda_fallback(size, stream);
    }
    break;

  case AllocationStrategy::PERFORMANCE_FIRST:
    // Prefer GPU allocation without fallback to preserve performance intent
    result.ptr = allocate_from_cuda(size);
    if (result.ptr) {
      result.status = Status::SUCCESS;
      result.allocated_size = size;
    } else {
      result.status = Status::ERROR_OUT_OF_MEMORY;
    }
    break;

  default:
    result = allocate_with_cuda_fallback(size, stream);
    break;
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  result.allocation_time_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end_time -
                                                           start_time)
          .count();

  update_allocation_latency(result.allocation_time_ns);

  // Register pointer for tracking (critical fix for deallocation)
  if (result.ptr) {
    std::lock_guard<std::mutex> m(pointer_map_mutex_);
    // Use pool_idx=-1 to indicate direct allocation (not from pool)
    pointer_index_map_[result.ptr] =
        PointerMeta{-1,   ++allocation_sequence_counter_, 0, nullptr,
                    size, result.is_host_memory};
  }

  return result;
}

FallbackAllocation
MemoryPoolManager::allocate_dual_memory(size_t size, cudaStream_t stream) {
  FallbackAllocation result;
  result.requested_size = size;
  result.is_dual_memory = true;

  // Allocate GPU memory
  result.ptr = allocate_from_cuda(size);

  // Allocate Host memory
  result.host_ptr = allocate_from_host(size);

  if (result.ptr && result.host_ptr) {
    result.status = Status::SUCCESS;
    result.allocated_size = size;
    dual_memory_allocations_.fetch_add(1, std::memory_order_relaxed);

    // Register GPU pointer for tracking (critical fix for deallocation)
    {
      std::lock_guard<std::mutex> m(pointer_map_mutex_);
      // Store both GPU and host pointers - GPU ptr is device memory
      pointer_index_map_[result.ptr] = PointerMeta{
          -1, ++allocation_sequence_counter_, 0, result.host_ptr, size, false};
    }
  } else {
    // If either failed, cleanup and return error
    if (result.ptr) {
      cudaFree(result.ptr); // Direct free since not registered yet
    }
    if (result.host_ptr) {
      cuda_zstd::util::debug_free(result.host_ptr);
      host_memory_usage_.fetch_sub(size, std::memory_order_relaxed);
    }
    result.ptr = nullptr;
    result.host_ptr = nullptr;
    result.status = Status::ERROR_OUT_OF_MEMORY;
  }

  return result;
}

// ============================================================================
// Resource Management
// ============================================================================

ResourceState MemoryPoolManager::get_current_resource_state() const {
  // If cached state is recent, return it
  if (std::chrono::steady_clock::now() - last_resource_update_ <
      std::chrono::milliseconds(10)) {
    return cached_resource_state_;
  }

  // Otherwise update (const cast needed for mutable cache)
  const_cast<MemoryPoolManager *>(this)->update_resource_state();
  return cached_resource_state_;
}

void MemoryPoolManager::update_resource_state() {
  ResourceState state;

  // GPU Memory
  size_t free_mem = 0, total_mem = 0;
  cudaMemGetInfo(&free_mem, &total_mem);
  state.available_gpu_memory = free_mem;
  state.total_system_memory = total_mem; // Approximate
  state.current_gpu_usage =
      current_memory_usage_.load(std::memory_order_relaxed);
  state.gpu_utilization =
      (total_mem > 0) ? (1.0f - (float)free_mem / total_mem) : 0.0f;

  // Host Memory (Tracking only what we allocated)
  state.current_host_usage = host_memory_usage_.load(std::memory_order_relaxed);
  state.available_host_memory =
      fallback_config_.host_memory_limit_mb * 1024 * 1024 -
      state.current_host_usage;

  // Pool Stats
  state.active_allocations =
      total_allocations_.load(std::memory_order_relaxed) -
      total_deallocations_.load(std::memory_order_relaxed);
  state.fragmentation_ratio = 0; // Simplified

  cached_resource_state_ = state;
  last_resource_update_ = std::chrono::steady_clock::now();
}

AllocationStrategy
MemoryPoolManager::select_optimal_strategy(const ResourceState &state,
                                           size_t size) const {
  if (state.gpu_utilization > 0.95f) {
    return AllocationStrategy::PREFER_HOST;
  }

  if (size > state.available_gpu_memory / 2) {
    return AllocationStrategy::PREFER_HOST;
  }

  if (state.gpu_utilization > 0.8f) {
    return AllocationStrategy::BALANCED;
  }

  return AllocationStrategy::PREFER_GPU;
}

size_t MemoryPoolManager::calculate_optimal_allocation_size(
    const ResourceState &state, size_t requested_size) const {
  // Simple implementation: if pressure is high, suggest smaller chunks
  if (state.gpu_utilization > 0.9f) {
    return requested_size / 2;
  }
  return requested_size;
}

Status MemoryPoolManager::perform_resource_balance() {
  // Intentionally a no-op: rebalancing (e.g. migrating host-fallback
  // allocations back to GPU when pressure drops) is deferred until
  // profiling shows it is a bottleneck.
  if (get_memory_pressure_percentage() < 50) {
    // Could trigger migration of host fallbacks to GPU
  }
  return Status::SUCCESS;
}

// ============================================================================
// Advanced Statistics
// ============================================================================

void MemoryPoolManager::get_detailed_statistics(PoolStats &stats) const {
  stats = get_statistics();
  // Add more detailed stats if PoolStats structure supports it
}

double MemoryPoolManager::get_average_allocation_latency() const {
  uint64_t count = allocation_latency_count_.load(std::memory_order_relaxed);
  if (count == 0)
    return 0.0;
  return (double)total_allocation_latency_ns_.load(std::memory_order_relaxed) /
         count;
}

void MemoryPoolManager::update_allocation_latency(uint64_t latency_ns) {
  total_allocation_latency_ns_.fetch_add(latency_ns, std::memory_order_relaxed);
  allocation_latency_count_.fetch_add(1, std::memory_order_relaxed);
}

// ============================================================================
// Global Pool Instance
// ============================================================================

static MemoryPoolManager *g_pool_instance = nullptr;
static std::mutex *g_pool_mutex = nullptr;
static std::once_flag g_pool_mutex_flag;

static std::mutex &get_pool_mutex() {
  std::call_once(g_pool_mutex_flag, []() {
    g_pool_mutex = new std::mutex();
  });
  return *g_pool_mutex;
}

MemoryPoolManager &get_global_pool() {
  std::lock_guard<std::mutex> lock(get_pool_mutex());
  if (!g_pool_instance) {
    g_pool_instance = new MemoryPoolManager(true);
  }
  return *g_pool_instance;
}

void destroy_global_pool() {
  std::lock_guard<std::mutex> lock(get_pool_mutex());
  if (g_pool_instance) {
    delete g_pool_instance;
    g_pool_instance = nullptr;
  }
}

} // namespace memory
} // namespace cuda_zstd
