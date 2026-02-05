# 常用操作<a name="ZH-CN_TOPIC_0000001698088057"></a>

## 日志说明<a name="ZH-CN_TOPIC_0000001506334653"></a>

检索日志组件基于《CANN 软件安装指南》以及《CANN 日志参考》设计和开发。

对于标准态部署，检索的日志属于应用类日志，可以参考《CANN 日志参考》中的“查看日志（Ascend EP标准形态）”章节的“查看应用类日志”描述。默认路径为“$HOME/ascend/log”。也可以使用环境变量ASCEND\_PROCESS\_LOG\_PATH指定日志落盘路径。命令参考如下：

```
export ASCEND_PROCESS_LOG_PATH=$HOME/xxx
```

可指定日志落盘路径为任意有读写权限的目录。

日志级别由低到高依次为DEBUG < INFO < WARNING < ERROR，级别越低，输出日志越详细，可以通过ASCEND\_GLOBAL\_LOG\_LEVEL环境变量设置日志级别。命令参考如下：

```
export ASCEND_GLOBAL_LOG_LEVEL=1
```

不传入此参数，默认为ERROR等级。ASCEND\_GLOBAL\_LOG\_LEVEL全部取值说明如下：

0：DEBUG

1：INFO

2：WARNING

3：ERROR

4：NULL，NULL级别。不输出日志。

> [!NOTE] 说明
>-   对于容器化场景中使用检索功能，应用类日志位于容器中，需要将日志目录挂载到宿主机才能实现持久化，否则日志将在容器退出时被销毁。
>-   应用类日志没有老化，日志会不断增多，因此需要用户定期清理该目录（可以使用系统自带的**logrotate**实现日志切分），否则可能导致磁盘空间不足，影响业务正常运行。
>-   软件包的安装升级卸载等管理面的相关日志会保存至“$HOME/log/mxIndex/deployment.log”，文件中保存有登录用户的用户名、访问端地址以及hostname，用于支持后续的日志记录及审计的操作。


