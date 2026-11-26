"""Publish verified report bytes without reopening a temporary path."""

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import os
from pathlib import Path, PureWindowsPath
from uuid import uuid4

from evaluation.portable_benchmark.pdf_output import directory_identity, verify_directory
from evaluation.portable_benchmark.ranking_report import validate_report


@dataclass(frozen=True)
class ReportDestination:
    path: Path
    directory_id: tuple[int, int]


def output_destination(path):
    absolute = Path(os.path.abspath(path))
    if (
        ".." in path.parts or absolute.drive.startswith("\\\\")
        or absolute.parent.resolve(strict=True) != absolute.parent
    ):
        raise ValueError("Choose an existing local report directory without redirection")
    expected = directory_identity(absolute.parent)
    if (
        absolute.suffix.lower() != ".json" or PureWindowsPath(absolute.name).is_reserved()
        or any(char in absolute.name for char in '<>:"/\\|?*')
        or any(ord(char) < 32 for char in absolute.name)
    ):
        raise ValueError("Choose a safe .json report filename")
    if os.path.lexists(absolute):
        raise FileExistsError("Output already exists. Choose a new report filename")
    evaluation = Path(__file__).resolve().parents[1]
    if any(parent in absolute.parents for parent in (evaluation / "results", evaluation / "datasets")):
        raise ValueError("Use a separate directory from historical reports and datasets")
    return ReportDestination(absolute, expected)


@contextmanager
def release_staging(release):
    """Preserve the primary failure if releasing staging also fails."""
    try:
        yield
    except BaseException as error:
        try:
            release()
        except Exception as cleanup_error:
            error.add_note(f"Report staging cleanup failed: {cleanup_error!r}")
        raise
    else:
        release()


@contextmanager
def posix_destination(output):
    destination = output.path
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    with ExitStack() as stack:
        directory = os.open(destination.anchor, flags)
        stack.callback(os.close, directory)
        for part in destination.parent.parts[1:]:
            directory = os.open(part, flags, dir_fd=directory)
            stack.callback(os.close, directory)
        info = os.fstat(directory)
        expected = output.directory_id
        if (info.st_dev, info.st_ino) != expected:
            raise OSError("Report output directory changed after preflight")
        verify_directory(destination.parent, expected)
        # Linux gives us an unnamed, exclusively owned inode. Failure or
        # interruption simply closes its last descriptor. There is no name
        # for another writer to replace, or for our cleanup to unlink.
        if not hasattr(os, "O_TMPFILE"):
            raise OSError("Anonymous report staging is unavailable on this platform")
        fd = os.open(".", os.O_TMPFILE | os.O_RDWR, 0o600, dir_fd=directory)
        # Keep ownership of the descriptor even if the stream constructor
        # creates and then destroys a partial wrapper.
        with release_staging(lambda: os.close(fd)):
            stream = os.fdopen(fd, "w+b", closefd=False)

            def commit():
                verify_directory(destination.parent, expected)
                # A destination dir_fd forces linkat, with AT_SYMLINK_FOLLOW.
                os.link(
                    f"/proc/self/fd/{fd}", destination.name,
                    dst_dir_fd=directory, follow_symlinks=True,
                )

            with release_staging(stream.close):
                yield stream, commit


@contextmanager
def windows_destination(output):
    destination = output.path
    import ctypes
    from ctypes import wintypes
    import msvcrt

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    ]
    create.restype = wintypes.HANDLE
    close = kernel.CloseHandle
    close.argtypes = [wintypes.HANDLE]
    close.restype = wintypes.BOOL
    set_info = kernel.SetFileInformationByHandle
    set_info.argtypes = [wintypes.HANDLE, wintypes.INT, wintypes.LPVOID, wintypes.DWORD]
    set_info.restype = wintypes.BOOL

    def open_handle(path, access, share, disposition, flags):
        handle = create(str(path), access, share, None, disposition, flags, None)
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        return handle

    def close_handle(handle):
        if not close(handle):
            raise ctypes.WinError(ctypes.get_last_error())

    def update(handle, kind, info):
        if not set_info(handle, kind, ctypes.byref(info), ctypes.sizeof(info)):
            code = ctypes.get_last_error()
            if code in (80, 183):
                raise FileExistsError("Output already exists. Choose a new report filename")
            raise ctypes.WinError(code)

    class RenameInfo(ctypes.Structure):
        _fields_ = [
            ("flags", wintypes.DWORD), ("root_directory", wintypes.HANDLE),
            ("name_length", wintypes.DWORD), ("name", wintypes.WCHAR * (len(str(destination)) * 2 + 1)),
        ]

    with ExitStack() as stack:
        # Pin every ancestor without FILE_SHARE_DELETE, and inspect reparse
        # points themselves rather than their targets. A pinned path is stable
        # for the handle-based rename on a local Windows filesystem.
        for parent in reversed(destination.parents):
            handle = open_handle(parent, 1, 3, 3, 0x02200000)
            stack.callback(close_handle, handle)
            directory_identity(parent)
        verify_directory(destination.parent, output.directory_id)
        temporary = destination.parent / f".aura-report-{uuid4().hex}.tmp"
        # READ | WRITE | DELETE, no sharing, CREATE_NEW, OPEN_REPARSE_POINT.
        handle = open_handle(temporary, 0xC0010000, 0, 1, 0x00200000)
        fd = None
        committed = False

        def release():
            # open_osfhandle transfers ownership to the CRT only on success.
            with ExitStack() as cleanup:
                if fd is None:
                    cleanup.callback(close_handle, handle)
                else:
                    cleanup.callback(os.close, fd)
                if not committed:
                    # Mark the file for deletion before its last handle closes.
                    update(handle, 4, wintypes.BOOL(True))

        def commit():
            nonlocal committed
            info = RenameInfo()
            # FILE_RENAME_INFO with replacement disabled and an absolute target.
            name = str(destination).encode("utf-16-le")
            info.name_length = len(name)
            ctypes.memmove(ctypes.addressof(info) + RenameInfo.name.offset, name, len(name))
            update(handle, 3, info)
            committed = True

        with release_staging(release):
            fd = msvcrt.open_osfhandle(handle, os.O_RDWR | os.O_BINARY)
            stream = os.fdopen(fd, "w+b", closefd=False)
            with release_staging(stream.close):
                yield stream, commit


def publish(report, destination, context):
    if not isinstance(destination, ReportDestination):
        destination = output_destination(destination)
    encoded = (report.model_dump_json(indent=2, exclude_none=True) + "\n").encode("utf-8")
    output = windows_destination if os.name == "nt" else posix_destination
    with output(destination) as (stream, commit):
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
        stream.seek(0)
        stored = stream.read()
        if stored != encoded:
            raise OSError("Stored report bytes differ from the serialized report")
        validate_report(stored, context)
        commit()
