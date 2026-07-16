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
