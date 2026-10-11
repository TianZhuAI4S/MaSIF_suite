"""Lightweight command dispatch, including hardware-aware runtime installation."""

import sys


def main(argv=None):
    """Set up the runtime before importing Torch, or dispatch the application CLI."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == 'install-runtime':
        from .runtime import main as setup
        setup(arguments[1:])
        # Console wrappers pass the return value to sys.exit; reports are printed, not exit codes.
        return None
    from .cli import main as application
    return application(arguments or ['--help'])
