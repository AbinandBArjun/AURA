from contextlib import ExitStack, contextmanager
import os
from pathlib import Path, PureWindowsPath
import re
import stat


def directory_identity(path):
    info = path.lstat()
    reparse_point = getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if not stat.S_ISDIR(info.st_mode) or reparse_point or not info.st_ino:
        raise OSError(f"Unsafe PDF output directory: {path}")
    return info.st_dev, info.st_ino


def verify_directory(path, expected):
    try:
        current = directory_identity(path)
    except OSError as error:
        raise OSError(f"PDF output directory changed or became unsafe: {path}") from error
    if current != expected:
        raise OSError(f"PDF output directory changed during generation: {path}")


def create_directory(directory, parent_fd=None):
    try:
        if parent_fd is None:
            directory.mkdir()
        else:
            os.mkdir(directory.name, mode=0o700, dir_fd=parent_fd)
    except FileExistsError as error:
        raise FileExistsError(f"PDF output directory already exists: {directory}") from error


@contextmanager
def posix_output(directory):
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    with ExitStack() as stack:
        parent_fd = os.open(directory.anchor, flags)
        stack.callback(os.close, parent_fd)
        for part in directory.parent.parts[1:]:
            try:
                parent_fd = os.open(part, flags, dir_fd=parent_fd)
            except OSError as error:
                raise OSError(f"Cannot open PDF output parent {directory.parent}: {error}") from error
            stack.callback(os.close, parent_fd)

        create_directory(directory, parent_fd)
        expected = directory_identity(directory)
        output_fd = os.open(directory.name, flags, dir_fd=parent_fd)
        stack.callback(os.close, output_fd)
        info = os.fstat(output_fd)
        if (info.st_dev, info.st_ino) != expected:
            raise OSError(f"PDF output directory changed during initialization: {directory}")

        def create_file(name):
            verify_directory(directory, expected)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
            fd = os.open(name, flags, mode=0o600, dir_fd=output_fd)
            return os.fdopen(fd, "wb")

        yield create_file
        verify_directory(directory, expected)


@contextmanager
def windows_output(directory):
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

    list_directory = 0x1
    generic_write = 0x40000000
    share_read_write = 0x3  # Omits FILE_SHARE_DELETE to block rename and deletion.
    open_existing = 3
    create_new = 1
    backup_semantics = 0x02000000
    open_reparse_point = 0x00200000
    invalid_handle = ctypes.c_void_p(-1).value

    def open_handle(path, access, share, disposition, flags):
        handle = create(str(path), access, share, None, disposition, flags, None)
        if handle == invalid_handle:
            error = ctypes.WinError(ctypes.get_last_error())
            error.filename = str(path)
            raise error
        return handle

    with ExitStack() as stack:
        def pin_directory(path):
            handle = open_handle(path, list_directory, share_read_write, open_existing,
                                 backup_semantics | open_reparse_point)
            stack.callback(close, handle)
            return directory_identity(path)

        for parent in reversed(directory.parents):
            pin_directory(parent)
        create_directory(directory)
        expected = directory_identity(directory)
        if pin_directory(directory) != expected:
            raise OSError(f"PDF output directory changed during initialization: {directory}")

        def create_file(name):
            verify_directory(directory, expected)
            handle = open_handle(directory / name, generic_write, 0, create_new, open_reparse_point)
            try:
                fd = msvcrt.open_osfhandle(handle, os.O_WRONLY | os.O_BINARY)
            except OSError:
                close(handle)
                raise
            return os.fdopen(fd, "wb")

        yield create_file
        verify_directory(directory, expected)


@contextmanager
def open_pdf_output(directory: Path):
    if not directory.name or ".." in directory.parts:
        raise ValueError("Choose a new PDF directory without '..' path components")
    directory = directory.absolute()
    output = windows_output if os.name == "nt" else posix_output
    with output(directory) as create_file:
        def open_pdf(name):
            if (
                not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*\.pdf", name)
                or PureWindowsPath(name).is_reserved()
            ):
                raise ValueError(f"Unsafe PDF filename: {name}")
            try:
                return create_file(name)
            except FileExistsError as error:
                raise FileExistsError(f"PDF destination already exists: {directory / name}") from error

        yield open_pdf
