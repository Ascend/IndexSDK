# Common Operations<a name="en-us_TOPIC_0000001698088057"></a>

## Log Description<a name="en-us_TOPIC_0000001506334653"></a>

The retrieval log component is developed based on the [CANN Software Installation Guide](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netyum) and [CANN Log Reference](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/maintenref/logreference/logreference_0001.html).

For standard mode deployments, retrieval logs are application logs. For details, see [Viewing Logs (Ascend EP)](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/maintenref/logreference/logreference_0002.html). The default path is `$HOME/ascend/log`. You can also use the `ASCEND_PROCESS_LOG_PATH` environment variable to specify the log storage path. The command is as follows:

```bash
export ASCEND_PROCESS_LOG_PATH=$HOME/xxx
```

You can specify any directory with read and write permissions as the log storage path.

The log levels, in ascending order, are DEBUG < INFO < WARNING < ERROR. The lower the level, the more detailed the output logs. You can use the `ASCEND_GLOBAL_LOG_LEVEL` environment variable to set the log level. The command is as follows:

```bash
export ASCEND_GLOBAL_LOG_LEVEL=1
```

If this parameter is not specified, the default log level is ERROR. The possible values of `ASCEND_GLOBAL_LOG_LEVEL` are as follows:

- `0`: DEBUG

- `1`: INFO

- `2`: WARNING

- `3`: ERROR

- `4`: NULL. No logs are output.

> [!NOTE]
>
> - When the retrieval function is used in containerized scenarios, application logs are stored inside the container. You must mount the log directory to the host to persist the logs. Otherwise, the logs are deleted when the container exits.
> - Application logs are not automatically rotated and continue to grow. Therefore, you need to periodically clean up the log directory. You can use the system-provided `logrotate` utility to rotate logs. Otherwise, insufficient disk space may affect normal service operation.
> - Logs related to management operations such as software package installation, upgrade, and uninstallation are saved to `$HOME/log/mxIndex/deployment.log`. This file contains the username of the logged-in user, the source address, and the hostname to support subsequent log recording and auditing.

## IVFRaBitQ Runtime Diagnostics<a name="ivfrabitq-runtime-debug"></a>

AscendIndexIVFRaBitQ provides three **optional debugging environment variables** for troubleshooting abnormal NPU uploads of coarse centroids and deviations in L1 coarse-search probe selection. **All are disabled by default** and have zero overhead on the production path. Enable them as needed only in development or integration testing environments.

> [!NOTE]
>
> - The environment variables are read **when the process starts**. You must export them **before** running the application or test case.
> - Diagnostic logs are output to stderr and to application logs when `IVFRABITQ_VERIFY_COARSE_CENTER` is enabled. It is recommended that you redirect the output to a `.log` file for easier use with `grep`.
> - `IVFRABITQ_VERIFY_L1_DIST` performs a full D2H transfer for L1 golden comparison. **Do not enable it continuously during performance benchmarking.**
> - After modifying the `RotateAndL2AtFP32` operator, recompile and deploy the custom OPP. Otherwise, the diagnostic results may still reflect the behavior of the old operator.

### Environment Variables

**Table 1** IVFRaBitQ debugging environment variables

| Environment Variable | Value | Trigger | Purpose |
| -- | -- | -- | -- |
| `IVFRABITQ_VERIFY_COARSE_CENTER` | Non-empty and not `0` | `copyFrom` / centroid update after training (`updateCoarseCenterImpl`) | Per-stage D2H verification to distinguish H2D memcpy failures from incomplete output from the rotate operator. |
| `IVFRABITQ_DEBUG_L1_PROBE` | `1` / `stats` / `full` | L1 stage of each `search` | Check whether probes are confined to `[0,8192)` and fail to cover the latter half of the lists. |
| `IVFRABITQ_VERIFY_L1_DIST` | Non-empty and not `0` | L1 stage of each `search` | Compare CPU golden results with NPU L1 distances and probes. |

To disable the debugging environment variables, run the following commands:

```bash
unset IVFRABITQ_VERIFY_COARSE_CENTER IVFRABITQ_DEBUG_L1_PROBE IVFRABITQ_VERIFY_L1_DIST
# Or set them to 0
export IVFRABITQ_VERIFY_COARSE_CENTER=0
```

### Diagnostic Decision Flow

1. **Low recall occurs after `copyFrom`** → Enable `IVFRABITQ_VERIFY_COARSE_CENTER=1` and rerun `copyFrom`.
2. Check `zeroRowsAfter2512` in the `centroidsOnDevice_rotated_full` log:
   - **`originCentroidsOnDevice` matches the host after H2D**, but `zeroRowsAfter2512 > 0` → The fault is in the `RotateAndL2AtFP32` operator.
   - A mismatch occurs immediately after H2D → Check the memcpy parameters or buffer capacity.
3. **Centroids are confirmed to be correct, but recall is still low** → Enable `IVFRABITQ_DEBUG_L1_PROBE=stats` and check the probe distribution in tile1 and tile2.
4. **L1 distances around the 8192 boundary need to be compared** → Enable `IVFRABITQ_VERIFY_L1_DIST=1` or `IVFRABITQ_DEBUG_L1_PROBE=full`.

### Diagnosing Coarse Center Upload

```bash
export IVFRABITQ_VERIFY_COARSE_CENTER=1

# Boundary test (copyFrom triggers updateCoarseCenterImpl)
./TestAscendIndexIVFRaBitQBoundary --gtest_filter=*CoarseCenterCopy10048* 2>&1 | tee coarse_verify.log

grep -E 'CoarseCenterVerify|zeroRowsAfter2512' coarse_verify.log
```

**Expected log after the fix:**

```text
[CoarseCenterVerify] originCentroidsOnDevice: row 2512 OK vs host (devNorm=...)
[CoarseCenterVerify] centroidsOnDevice_rotated_full full: mismatchRows=0 zeroRowsBefore2512=0 zeroRowsAfter2512=0 / 7536
```

**Typical abnormal log before the fix (`nlist=10048`):**

```text
[CoarseCenterVerify] centroidsOnDevice_rotated: row 2512 is all-zero on device (devNorm=0.000000)
[CoarseCenterVerify] centroidsOnDevice_rotated_full full: zeroRowsAfter2512=7536 / 7536
```

**Key finding:** `2512 = 10048 / 4`, which is the per-core batch size when four AIC cores are evenly assigned. Row 2512 is the first boundary row outside core 0.

### Diagnosing L1 Probe Distribution

```bash
# Print the first 8 probe IDs for q0
export IVFRABITQ_DEBUG_L1_PROBE=1

# Only collect probe distribution statistics for [0,8192) and [8192,nlist) (recommended; low overhead)
export IVFRABITQ_DEBUG_L1_PROBE=stats

# Probe list + CPU golden comparison + distribution statistics (highest overhead)
export IVFRABITQ_DEBUG_L1_PROBE=full
```

**Example in `stats` mode:**

```text
[IVFRaBitQ] L1 probe stats q0: nprobe=1024 in[0,8192)=1024 in[8192,10048)=0 min=3 max=8191
```

Abnormal signal: `in[8192,nlist)=0` when `nlist > 8192` indicates that the probes do not cover the latter half of the lists. This is often related to zero rows in the device-side centroids or an abnormal L1 distance operator.

### Comparing L1 Distances with Golden Results

```bash
export IVFRABITQ_VERIFY_L1_DIST=1
# Run the search scenario
grep 'L1 dist golden\|jaccard' search.log
```

**Sampled boundary IDs (8191/8192/8193 and tail rows):**

```text
[IVFRaBitQ] L1 dist golden id=8192 cpu=15.678901 npu=15.678902 absErr=0.000001
```

**Probe overlap:**

```text
[IVFRaBitQ] L1 probe overlap q0: nprobe=1024 overlap=980 jaccard=0.957 cpu_tile2=128
```

| Field       | Description                                   | Healthy Reference                                                      |
| ----------- | --------------------------------------------- | ---------------------------------------------------------------------- |
| `overlap`   | Intersection of CPU top-nprobe and NPU probes | Close to `nprobe`                                                      |
| `jaccard`   | `overlap / nprobe`                            | > 0.95                                                                 |
| `cpu_tile2` | Number of CPU golden probes in `[8192,nlist)` | If `NPU tile2=0` while `cpu_tile2>0`, the NPU is missing the latter half. |

### Recommended Combinations

| Scenario                               | Export combination                 |
| -------------------------------------- | ---------------------------------- |
| Verify `copyFrom` upload               | `IVFRABITQ_VERIFY_COARSE_CENTER=1` |
| Centroids are correct; troubleshoot L1 | `IVFRABITQ_DEBUG_L1_PROBE=stats`   |
| L1 distances around the 8192 boundary  | `IVFRABITQ_VERIFY_L1_DIST=1`       |
| Perform a complete dump in one run     | `IVFRABITQ_DEBUG_L1_PROBE=full`    |

### Further Reading

- [FAQ: IVFRaBitQ Retrieval Recall Issues](./07_faq.md#ivfrabitq-recall-low-nlist-10048)

## IVFRaBitQ Distance Operator Block Layout<a name="ivfrabitq-topk-layout"></a>

The order in which the L2 distance operator lays its blocks out on the core grid can be switched with environment variables. Unlike the diagnostic switches above, **this group enables the new (`group`) layout by default**. Set `IVFRABITQ_TOPK_LAYOUT=legacy` to restore the historical query-major contiguous layout, which serves as the control arm for A/B comparison. For a summary of the variables, see [Appendix](./09_appendix.md#ivfrabitq-topk-layout-env).

The historical layout fills the grid query-major: all blocks of query 0, then query 1, and so on. Later queries start late and therefore finish late, so the TopK side waits in a staircase pattern. The `group` layout instead round-robins over several adjacent queries within a window, running the windows back to back, so a window's queries finish at roughly the same time and the next window starts on the threads that just freed up.

> [!NOTE]
>
>- Both variables are read **once when the `AscendIndexIVFRaBitQ` index is constructed** and cached. They are **not** re-read on each `search`. Export them **before** creating the index; changing them at runtime has no effect.
>- The two layouts only change where blocks land on the grid, not the data that takes part in the computation. The TopK output is bit-for-bit identical, and `group=1` reproduces exactly the `legacy` fill order.
>- This optimization changes the operator input count from 7 to 8, so **`TopkIvfRabitqFp32.om` must be regenerated** (`aicpu_generate_model.py` is already updated). Reusing an old `.om` fails the count check with `KERNEL_STATUS_PARAM_INVALID` rather than failing silently.

### Environment Variables

**Table 2** IVFRaBitQ block layout environment variables

|Environment Variable|Values|Default|Purpose|
|--|--|--|--|
|IVFRABITQ_TOPK_LAYOUT|`legacy` or `group` (`grouped` is equivalent)|`group`|Selects the block layout. `legacy` is the historical query-major contiguous layout, kept only as the A/B baseline|

An invalid value **falls back to the default silently, with no WARNING**: any string other than `legacy`/`group`/`grouped` is treated as the default `group`. To confirm which value actually took effect, rely on A/B measurements.

The window width (how many adjacent queries one window holds) **needs no configuration; it is queried from the device at runtime.** What actually constrains it is how many of a window's queries the TopK operator can drain in parallel, so widening the window beyond that only delays every query's finish time. At index construction the host reads the device's AI CPU count through ACL's `aclrtGetDeviceCapability(ACL_DEV_ATTR_AICPU_CORE_NUM)` and uses it as the window width. Under the `legacy` layout the width is forced to `1`.

If the constructing thread has not run `aclrtSetDevice` yet the query fails; the constant `6` stands in and the **first search** retries once, logging an INFO line with the width that ended up in effect. Building the index before setting the device therefore does not pin the fallback for good. If the device never answers (a CANN version without the attribute, a simulator), the constant `6` stays in use.

> [!NOTE]
>ACL reports the **device's** AI CPU count, whereas the operator sees the count granted to **this context** (`CpuKernelUtils::GetCPUNum`, which it already mins with `nq`). On a card shared by several index instances the former reads high. This costs parallelism but never correctness — `blockSlots` is an explicit bijection the operator reads back, so the window width never changes which blocks a query scans; a too-wide window at worst degenerates to the per-block round-robin that `legacy` already represents.

To restore the historical layout:

```bash
# Control arm; the window width is forced to 1 here
export IVFRABITQ_TOPK_LAYOUT=legacy

# Restore the default (group layout, window width taken from the device AI CPU count)
unset IVFRABITQ_TOPK_LAYOUT
```

### Examples

```bash
# Control arm: historical query-major layout
export IVFRABITQ_TOPK_LAYOUT=legacy
./your_benchmark 2>&1 | tee layout_legacy.log

# Experiment arm: default group layout, window width taken from the device AI CPU count
unset IVFRABITQ_TOPK_LAYOUT
./your_benchmark 2>&1 | tee layout_group.log
```
