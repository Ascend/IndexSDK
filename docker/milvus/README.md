# Milvus images for A2 and A3

This Dockerfile keeps the published `milvus:v2.5.19-ascend-arm` runtime and
replaces its CANN/IndexSDK stack with CANN 9.1.0 and IndexSDK 26.1.0. The
published Milvus image supplies the Milvus binary and its native runtime
libraries; the CANN image supplies the target-chip toolkit.

Download the aarch64 IndexSDK package into this directory before building:

```bash
curl -fL \
  -o Ascend-mindxsdk-mxindex_26.1.0_linux-aarch64.run \
  https://gitcode.com/Ascend/IndexSDK/releases/download/v26.1.0/Ascend-mindxsdk-mxindex_26.1.0_linux-aarch64.run
chmod 550 Ascend-mindxsdk-mxindex_26.1.0_linux-aarch64.run

# Use the SHA-256 published on the official release page; do not invent a value.
INDEXSDK_SHA256='<OFFICIAL_RELEASE_SHA256>'
printf '%s  %s\n' "$INDEXSDK_SHA256" \
  Ascend-mindxsdk-mxindex_26.1.0_linux-aarch64.run | sha256sum -c -
```

Build the A2 image:

```bash
docker build \
  -f Dockerfile.ubuntu \
  -t milvus:v2.5.19-ascend-26.1.0-cann9.1.0-910b-ubuntu22.04-py3.12 .
```

Build the A3 image:

```bash
docker build \
  -f Dockerfile.ubuntu \
  --build-arg CANN_VARIANT=a3 \
  --build-arg PLATFORM=A3 \
  --build-arg HARDWARE=A3 \
  -t milvus:v2.5.19-ascend-26.1.0-cann9.1.0-a3-ubuntu22.04-py3.12 .
```

The CANN and Milvus base images must be pulled from SWR. Docker Hub is not a
build dependency. At runtime, mount the host driver and NPU devices as shown
in `docker/OVERVIEW.md`, then run `./bin/milvus run standalone` from
`/home/milvus`.

## Runtime user

The image runs explicitly as `root`, matching the published Milvus base image.
The base image has no `milvus` account, and Ascend NPU access depends on the
permissions of host-provided device nodes. A deployment that requires
non-root operation must create the runtime account, grant it ownership of the
writable Milvus data directories, and add it to the host group owning
`/dev/davinci*`, `/dev/davinci_manager`, `/dev/hisi_hdc`, or the equivalent
device nodes configured by the deployment. This repository does not claim a
non-root device-permission setup has been validated.
