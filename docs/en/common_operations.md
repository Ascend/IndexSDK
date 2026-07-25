# Common Operations<a name="ZH-CN_TOPIC_0000001698088057"></a>

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-05-30T02:41:18.086Z -->

## Log Description<a name="ZH-CN_TOPIC_0000001506334653"></a>

The retrieval log component is developed based on the *CANN Software Installation Guide* and the *CANN Log Reference*.

In standard mode deployments, retrieval logs are application logs. You can refer to the "Viewing Application Logs" description in the "Viewing Logs (Ascend EP Standard Form)" chapter of the *CANN Log Reference*. The default path is `$HOME/ascend/log`. You can also use the `ASCEND_PROCESS_LOG_PATH` environment variable to specify the log storage path. The command reference is as follows:

```bash
export ASCEND_PROCESS_LOG_PATH=$HOME/xxx
```

You can specify the log storage path as any directory with read and write permissions.

The log levels from low to high are DEBUG < INFO < WARNING < ERROR. The lower the level, the more detailed the output. You can set the log level with the `ASCEND_GLOBAL_LOG_LEVEL` environment variable. The command reference is as follows:

```bash
export ASCEND_GLOBAL_LOG_LEVEL=1
```

If you do not pass this parameter, the default level is ERROR. The possible values of `ASCEND_GLOBAL_LOG_LEVEL` are as follows:

`0`: DEBUG

`1`: INFO

`2`: WARNING

`3`: ERROR

`4`: NULL. No logs are output.

> [!NOTE]
>
>- When you use the retrieval function in containerized scenarios, application logs are located inside the container. You need to mount the log directory on the host machine to ensure persistence. Otherwise, the logs are destroyed when the container exits.
>- Application logs do not age out, so they continue to grow. Therefore, you need to periodically clean up this directory. You can use the built-in `logrotate` of the system to implement log rotation. Otherwise, insufficient disk space may occur, affecting normal service operation.
>- Logs related to software package installation, upgrade, and uninstallation in the management plane are saved to `$HOME/log/mxIndex/deployment.log`. The file stores the username of the logged-in user, the source address, and the hostname, which are used to support subsequent log recording and auditing operations.

## Device Memory Debugging<a name="ascendfaiss-mem-debug"></a>

AscendFaiss provides two **optional** environment variables for diagnosing Device HBM allocation failures and HBM usage changes during index upload or list growth. They are **disabled by default** with zero overhead on the production path. Enable them only in development or integration environments. See also [Appendix](./appendix.md#ascendfaiss-mem-debug-env).

> [!NOTE]
>
>- Environment variables are read and cached on the **first query in the process**. Export them **before** starting the application or test.
>- Debug logs go to stderr and are also written to APP logs (prefix `[MemDebug]`). Redirect to a `.log` file for easier grepping.
>- When enabled, the code queries `aclrtGetMemInfo(ACL_HBM_MEM)` and samples allocations. **Do not leave this on during performance benchmarks.**

### Environment Variables

**Table 1** Device memory debug environment variables

|Environment Variable|Value|Default|Purpose|
|--|--|--|--|
|ASCENDFAISS\_MEM\_DEBUG|Non-empty and not `0` / `false` / `off` (case-insensitive)|Off|Master switch: sample allocations and HBM free/total; on `aclrtMalloc` failure, dump the recent allocation ring buffer|
|ASCENDFAISS\_MEM\_DEBUG\_EVERY|Positive integer N|`64`|Sampling period: print when `seq % N == 0`, or when `size ≤ 4096`; unset/invalid/0 falls back to `64`|

Disable:

```bash
unset ASCENDFAISS_MEM_DEBUG ASCENDFAISS_MEM_DEBUG_EVERY
# or
export ASCENDFAISS_MEM_DEBUG=0
```

### Usage Example

```bash
# Enable memory debugging (sample every 64 allocations by default)
export ASCENDFAISS_MEM_DEBUG=1

# Increase sampling density (print every 8th allocation)
export ASCENDFAISS_MEM_DEBUG=1
export ASCENDFAISS_MEM_DEBUG_EVERY=8

# Run the workload or UT and capture logs
./your_app 2>&1 | tee mem_debug.log
grep '\[MemDebug\]' mem_debug.log
```

### Log Interpretation

|Log keyword|Meaning|
|--|--|
|`[MemDebug] alloc seq=...`|Sampled Device allocation: sequence, size, space, device, HBM free/total before alloc|
|`[MemDebug] ... HBM free=...`|HBM free/total snapshot on a key path (for example `copyVectorToDevice_*`, `DeviceMemArena::Grow`, `IndexIVFRaBitQ_resize`)|
|`[MemDebug] aclrtMalloc FAILED ...`|Allocation failure: requested size/space/device, error code, HBM free/total at failure time|
|`dumping last N alloc records`|Up to 64 most recent allocations before failure (oldest→newest), used to locate the allocation spike|

**Typical workflow:**

1. Set `ASCENDFAISS_MEM_DEBUG=1` before reproducing OOM / `aclrtMalloc` failure.
2. Check whether `HBM_free` is near zero and note `size` / `space` (`DEVICE` or `DEVICE_HUGEPAGE`).
3. Use the dumped recent allocation records to correlate with the business stage (index upload, list resize, arena grow).
4. For denser sampling: `export ASCENDFAISS_MEM_DEBUG_EVERY=1` (logs every allocation; very verbose).
