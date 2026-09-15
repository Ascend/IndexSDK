# Security Hardening<a name="en-us_TOPIC_0000001506495801"></a>

## Security Requirements<a name="en-us_TOPIC_0000001456695036"></a>

When you use an API to read a file, ensure that you own the file and that its permissions are no more permissive than `640`. This helps prevent privilege escalation and similar security issues.

Software code or programs downloaded from external sources may pose security risks. You are responsible for ensuring the security of their functions.

## Hardening Precautions<a name="en-us_TOPIC_0000001550831169"></a>

The security hardening measures listed in this document provide basic recommendations. You should re-evaluate the network security hardening measures for the entire system based on your specific service requirements. When necessary, refer to industry best practices and consult security experts.

## OS Security Hardening<a name="en-us_TOPIC_0000001456854844"></a>

### Firewall Configuration<a name="en-us_TOPIC_0000001499591392"></a>

After installing the OS, if common users are configured, you can add `ALWAYS_SET_PATH=yes` to the `/etc/login.defs` file to prevent unauthorized privilege escalation.

### Setting umask<a name="en-us_TOPIC_0000001499751300"></a>

Set the umask to `027` or more restrictive on the host and in containers to enhance file security.

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

### Security Hardening for Ownerless Files<a name="en-us_TOPIC_0000001499431524"></a>

Differences between official Docker images and the host OS may result in mismatched user and group definitions. This can cause files created during operation on the host or in containers to become ownerless.

You can find ownerless files on the host or in containers by running `find / -nouser -o -nogroup`. Based on the UIDs and GIDs of these files, create the corresponding users and groups, or modify the UIDs and GIDs of existing users and groups to match. This ensures that the files have valid owners and prevents ownerless files from posing security risks to the system.

### Port Scanning<a name="en-us_TOPIC_0000001550471229"></a>

Pay attention to ports listening on all interfaces and unnecessary ports, and close unnecessary ports promptly. You are advised to disable insecure services, such as Telnet and FTP. For details about how to disable these services, see the documentation for the OS in use.

### Anti-DoS Protection<a name="en-us_TOPIC_0000001550671317"></a>

You can protect the system against Denial of Service (DoS) attacks by restricting the connection rate from each IP address. Methods include, but are not limited to, using the Linux `iptables` firewall and optimizing `sysctl` parameters. For details about how to use these methods, see the relevant documentation.

## Retrieval Service Security Hardening<a name="en-us_TOPIC_0000001742120313"></a>

**Proper Memory Planning<a name="section13209332319"></a>**

Plan memory usage properly to ensure that it does not exceed system resource limits. The retrieval service's feature base library is stored in the DDR memory of the Ascend AI Processor. The feature dimensions and data volume for operations such as indexing and querying, together with the temporary memory used by the service and the system during computation, determine the total memory footprint. Excessive input may cause device-side memory allocation failures. The maximum capacity supported by a single `Index` instance depends on the amount of device-side memory available on the specific Ascend AI Processor. Plan the number of `Index` instances based on actual requirements to prevent the memory limit from being exceeded.

**OMP Settings<a name="section62161855233"></a>**

If you need to modify OMP-related configurations, evaluate system resource limits such as memory and thread count. Otherwise, exceptions may occur during operation. For example, you can control the concurrency level by setting the `${OMP_NUM_THREADS}` environment variable. For details about OMP settings, see the official OMP guide.

**Interface Usage<a name="section8919813343"></a>**

Most retrieval interfaces use C-style input parameters. Therefore, you must ensure that each input pointer references a valid memory range of the required length. Otherwise, exceptions may occur during operation.

**Mutual Conversion with `faiss::Index`<a name="section943033715416"></a>**

The retrieval service provides mutual conversion with `faiss::Index`. Ensure that the `faiss::Index` output by `copyTo` is not modified. Otherwise, `copyFrom` may fail. The same applies to interfaces such as `index_ascend_to_cpu`, `index_int8_ascend_to_cpu`, `index_cpu_to_ascend`, and `index_int8_cpu_to_ascend`.
