# Security Hardening<a name="ZH-CN_TOPIC_0000001506495801"></a>

## Security Requirements<a name="ZH-CN_TOPIC_0000001456695036"></a>

When you use an API to read a file, ensure that you own the file and that its permissions are no more permissive than `640`. This helps prevent privilege escalation and similar security issues.

Software code or programs downloaded from external sources may pose risks. You must guarantee the security of their functions.

## Hardening Precautions<a name="ZH-CN_TOPIC_0000001550831169"></a>

The security hardening measures listed in this document provide basic recommendations. You should re-evaluate the network security posture of the entire system based on specific service requirements. When necessary, consult industry best practices and security experts.

## OS Security Hardening<a name="ZH-CN_TOPIC_0000001456854844"></a>

### Firewall Configuration<a name="ZH-CN_TOPIC_0000001499591392"></a>

After installing the OS, if common users are configured, you can add `ALWAYS_SET_PATH yes` to the `/etc/login.defs` file to prevent unauthorized privilege escalation.

### Setting umask<a name="ZH-CN_TOPIC_0000001499751300"></a>

Set the host umask to `027` or more restrictive on the host and in containers to enhance file security.

To set umask to `027`:

1. Log in to the server as the root user and edit the `/etc/profile` file.

    ```bash
    vim /etc/profile
    ```

2. Add `umask 027` to the end of the `/etc/profile` file, then save and exit.
3. Run the following command to apply the configuration.

    ```bash
    source /etc/profile
    ```

### Ownerless File Hardening<a name="ZH-CN_TOPIC_0000001499431524"></a>

Differences between official Docker images and the host OS may result in a mismatch between user definitions. This can lead to the creation of ownerless files during system or container operation.

You can find ownerless files on the host or in containers by running `find / -nouser -o -nogroup`. To mitigate security risks, create corresponding users and groups based on file UIDs and GIDs, or adjust existing UIDs and GIDs to match, thereby ensuring every file has a valid owner.

### Port Scanning<a name="ZH-CN_TOPIC_0000001550471229"></a>

Monitor ports listening on all interfaces and identify unnecessary ports for immediate closure. You are advised to disable insecure services, such as Telnet and FTP. For details, see the related documents of the OS in use.

### Anti-DoS Protection<a name="ZH-CN_TOPIC_0000001550671317"></a>

Protect the system against Denial of Service (DoS) attacks by implementing IP address restriction and rate limiting. Recommended methods include using the Linux iptables firewall and optimizing sysctl parameters. For details, see related documents.

## Retrieval Service Security Hardening<a name="ZH-CN_TOPIC_0000001742120313"></a>

**Proper memory planning<a name="section13209332319"></a>**

Users need to plan memory usage properly to ensure that it does not exceed system resource limits. Additionally, the retrieval service's feature base library is stored in DDR memory of the Ascend AI Processor. The feature dimensions and quantity during operations such as enrollment or query, as well as the use of temporary business memory and temporary system memory during computation, determine the total memory footprint. Excessive input can lead to device-side memory allocation failure errors. Currently, the maximum capacity supported by a single `Index` instance depends on the specific device-side memory size of the Ascend AI Processor. The service side needs to plan the number of `Index` instances based on actual requirements to prevent memory overrun scenarios.

**OMP settings<a name="section62161855233"></a>**

If you need to modify OMP-related configurations, evaluate system resource limits such as memory and thread count. Otherwise, exceptions may occur during operation. For example, you can set the concurrency level by setting `${OMP_NUM_THREADS}`. For details about OMP settings, see the official OMP guide.

**Interface usage<a name="section8919813343"></a>**

Most retrieval interfaces use C-style input parameters. Therefore, you must ensure that the length of the input pointer is a valid value. Otherwise, exceptions may occur during operation.

**Mutual conversion with `faiss::Index`<a name="section943033715416"></a>**

The retrieval service provides mutual conversion with `faiss::Index`. Ensure that the `faiss::Index` output by `copyTo` is not modified. Otherwise, it may cause `copyFrom` exceptions. The same applies to interfaces such as `index_ascend_to_cpu`, `index_int8_ascend_to_cpu`, `index_cpu_to_ascend`, and `index_int8_cpu_to_ascend`.
