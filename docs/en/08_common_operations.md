# Common Operations<a name="en-us_TOPIC_0000001698088057"></a>

## Log Description<a name="en-us_TOPIC_0000001506334653"></a>

The retrieval log component is developed based on the [CANN Software Installation Guide](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/softwareinst/instg/instg_0000.html) and [CANN Log Reference](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/maintenref/logreference/logreference_0001.html).

For standard mode deployments, retrieval logs are application logs. For details, see [Viewing Logs (Ascend EP)](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/maintenref/logreference/logreference_0002.html). The default path is `$HOME/ascend/log`. You can also use the `ASCEND_PROCESS_LOG_PATH` environment variable to specify the log storage path. The command is as follows:

```bash
export ASCEND_PROCESS_LOG_PATH=$HOME/xxx
```

You can specify any directory with read and write permissions as the log storage path.

The log levels, in ascending order, are DEBUG < INFO < WARNING < ERROR. The lower the level, the more detailed the output logs. You can use the `ASCEND_GLOBAL_LOG_LEVEL` environment variable to set the log level. The command is as follows:

```bash
export ASCEND_GLOBAL_LOG_LEVEL=1
```

If this parameter is not specified, the default log level is ERROR. The possible values of `ASCEND_GLOBAL_LOG_LEVEL` are as follows:

`0`: DEBUG

`1`: INFO

`2`: WARNING

`3`: ERROR

`4`: NULL. No logs are output.

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
./TestAscendIndexIVFRaBitQBoundary --gtest_filter="*CoarseCenterCopy10048*" 2>&1 | tee coarse_verify.log

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

## Device Memory Debugging<a name="ascendfaiss-mem-debug"></a>

AscendFaiss provides two **optional** environment variables for troubleshooting Device HBM allocation failures and changes in memory usage during index upload or expansion. **Both are disabled by default** and have zero overhead on the production path. Enable them as needed only in development or integration testing environments. For a list of the environment variables, see the [Appendix](./09_appendix.md#ascendfaiss-mem-debug-env).

> [!NOTE]
>
> - The environment variables are read and cached **the first time the process queries them**. You must export them **before** running the application or test case.
> - Debugging logs are output to stderr and are also written to application logs (with the `[MemDebug]` prefix). You are advised to redirect the output to a `.log` file for easier use with `grep`.
> - When the master switch is enabled, AscendFaiss queries `aclrtGetMemInfo(ACL_HBM_MEM)` and prints sampled results. **Do not enable it continuously during performance benchmarking.**

### Environment Variables

**Table 2** Device memory debugging environment variables

<table><tbody>
<tr><td align="center" valign="middle"><strong>Environment Variable</strong></td><td align="center" valign="middle"><strong>Value</strong></td><td width="70" align="center" valign="middle"><strong>Default</strong></td><td align="center" valign="middle"><strong>Purpose</strong></td></tr>
<tr><td valign="middle">ASCENDFAISS_MEM_DEBUG</td><td valign="middle">Non-empty and not <code>0</code> / <code>false</code> / <code>off</code> (case-insensitive)</td><td width="70" align="center" valign="middle">Disabled</td><td valign="middle">Master switch. Enables allocation sampling and HBM headroom printing. Dumps a ring buffer of the most recent allocations when <code>aclrtMalloc</code> fails.</td></tr>
<tr><td valign="middle">ASCENDFAISS_MEM_DEBUG_EVERY</td><td valign="middle">Positive integer N</td><td width="70" align="center" valign="middle"><code>64</code></td><td valign="middle">Sampling period. Prints allocation details when the allocation sequence number satisfies <code>seq % N == 0</code>, or when <code>size ≤ 4096</code>. Falls back to <code>64</code> when this variable is unset, invalid, or <code>0</code>.</td></tr>
</tbody></table>

To disable the debugging environment variables, run the following commands:

```bash
unset ASCENDFAISS_MEM_DEBUG ASCENDFAISS_MEM_DEBUG_EVERY
# Or
export ASCENDFAISS_MEM_DEBUG=0
```

### Usage Example

```bash
# Enable memory debugging (samples every 64th allocation by default)
export ASCENDFAISS_MEM_DEBUG=1

# Increase the sampling density (prints every 8th allocation)
export ASCENDFAISS_MEM_DEBUG=1
export ASCENDFAISS_MEM_DEBUG_EVERY=8

# Run the service or unit tests and save the logs
./your_app 2>&1 | tee mem_debug.log
grep '\[MemDebug\]' mem_debug.log
```

### Log Interpretation

|Log Keyword|Meaning|
|--|--|
|`[MemDebug] alloc seq=...`|A sampled record of one Device allocation: the sequence number, size, space, device, and HBM free/total before the allocation.|
|`[MemDebug] ... HBM free=...`|A snapshot of the HBM headroom on critical paths (such as `copyVectorToDevice_*`, `DeviceMemArena::Grow`, and `IndexIVFRaBitQ_resize`).|
|`[MemDebug] aclrtMalloc FAILED ...`|An allocation failure: the requested size, space, and device, the error code, and the HBM free/total at the time of failure.|
|`dumping last N alloc records`|Up to the 64 most recent allocation records before the failure (from oldest to newest), used to identify the consecutive allocations that exhaust Device memory.|

**Typical troubleshooting process:**

1. Before reproducing an OOM or `aclrtMalloc` failure, set `ASCENDFAISS_MEM_DEBUG=1`.
2. In the failure log, check whether `HBM_free` is close to 0, and check `size` and `space` (`DEVICE` or `DEVICE_HUGEPAGE`).
3. Use the most recent allocation records in the dump to correlate them with the service phase (index upload, list resize, or arena grow) and narrow down the leak or the peak allocation point.
4. To sample more densely, increase the frequency: `export ASCENDFAISS_MEM_DEBUG_EVERY=1` (every allocation is printed, which produces a large volume of logs).
