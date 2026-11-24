"""Issue Qdrant HTTP requests through exec in one verified Linux container instance."""

from dataclasses import dataclass
from http.client import HTTPResponse
from io import BytesIO
import re
import subprocess

import httpx


INSTANCE_PATTERN = r"pid:\[[0-9]+\]:[0-9]+"
# The pinned image runs a Bash entrypoint as PID 1. The PID namespace and its
# start time both change when Docker restarts the same container.
INSTANCE_SCRIPT = r'''
read -r stat </proc/1/stat || exit 46
read -r -a process_fields <<< "${stat##*) }"
namespace=$(readlink /proc/1/ns/pid) || exit 46
instance="$namespace:${process_fields[19]}"
'''
FORWARD_SCRIPT = r'''
expected=$1
''' + INSTANCE_SCRIPT + r'''
[[ "$instance" == "$expected" ]] || exit 47
exec 3<>/dev/tcp/127.0.0.1/6333 || exit 48
head -c "$2" >&3 || exit 48
cat <&3
'''


class ContainerTransportError(ValueError):
    """Transport error with a safe, credential-free public message."""


def exec_qdrant(context, container_id, script, *arguments, data=b""):
    try:
        result = subprocess.run(
            [
                "docker", "--context", context, "exec", "-i", "-e", "BASH_ENV=",
                container_id, "/bin/bash", "--noprofile", "--norc", "-c", script,
                "aura-benchmark-qdrant", *arguments,
            ],
            input=data, capture_output=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        raise ContainerTransportError("Qdrant container transport failed or timed out") from None
    if result.returncode == 47:
        raise ContainerTransportError("Qdrant process identity changed inside the inspected container")
    if result.returncode != 0:
        raise ContainerTransportError("The inspected Qdrant container or its loopback service is unavailable")
    return result.stdout


def read_instance(context, container_id):
    token = exec_qdrant(context, container_id, INSTANCE_SCRIPT + '\nprintf "%s" "$instance"\n').decode("ascii")
    if not re.fullmatch(INSTANCE_PATTERN, token):
        raise ContainerTransportError("The inspected Qdrant process identity could not be verified")
    return token


@dataclass(frozen=True)
class ResponseBuffer:
    data: bytes

    def makefile(self, mode):
        return BytesIO(self.data)


class ContainerQdrantTransport(httpx.BaseTransport):
    """No host TCP connection or reconnection to a replacement container."""

    def __init__(self, context, container_id, instance):
        self.context = context
        self.container_id = container_id
        self.instance = instance

    def handle_request(self, request):
        if (
            request.url.scheme != "http" or request.url.host != "127.0.0.1"
            or request.url.port != 6333
        ):
            raise ContainerTransportError("Qdrant transport only accepts the container loopback origin")
        body = request.read()
        headers = [
            (name, value) for name, value in request.headers.raw
            if name.lower() not in (b"accept-encoding", b"connection", b"content-length", b"transfer-encoding")
        ]
        headers.extend((
            (b"Accept-Encoding", b"identity"),
            (b"Connection", b"close"),
            (b"Content-Length", str(len(body)).encode("ascii")),
        ))
        first = request.method.encode("ascii") + b" " + request.url.raw_path + b" HTTP/1.1"
        data = b"\r\n".join([first, *(name + b": " + value for name, value in headers), b"", body])
        raw = exec_qdrant(self.context, self.container_id, FORWARD_SCRIPT, self.instance, str(len(data)), data=data)
        response = HTTPResponse(ResponseBuffer(raw), method=request.method)
        try:
            response.begin()
            content = response.read()
            headers = [(name, value) for name, value in response.getheaders() if name.lower() != "transfer-encoding"]
            return httpx.Response(response.status, headers=headers, content=content, request=request)
        finally:
            response.close()
