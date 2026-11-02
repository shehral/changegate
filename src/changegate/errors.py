"""Exception types. Every error changegate raises on purpose derives from ChangegateError."""

from __future__ import annotations

from pathlib import Path


class ChangegateError(Exception):
    """Base class for errors raised by changegate.

    Does not establish: anything about the model under test; errors describe the tool's run.
    """


class ConfigError(ChangegateError):
    """gate.yaml is unreadable or does not match the schema.

    Does not establish: that a config which loads is a sensible envelope for your change.
    """


class ArtifactError(ChangegateError):
    """An artifact is missing, has an unexpected digest, or is in a disallowed format.

    Does not establish: tampering; a digest mismatch can also be a partial download.
    """


class LogChainError(ChangegateError):
    """The hash-chained action log does not verify.

    Attributes:
        path: the log file.
        line: the 1-indexed line where verification stopped.

    Does not establish: who edited the log, or that an intact chain was never rewritten
    wholesale; compare the head with one recorded off the machine.
    """

    def __init__(self, path: Path, line: int, reason: str) -> None:
        """Build the error.

        Args:
            path: the log file.
            line: the 1-indexed line where verification stopped.
            reason: a short description of the failure.

        Does not establish: anything beyond the first broken line.
        """
        super().__init__(f"{path}: line {line}: {reason}")
        self.path = path
        self.line = line


class NotBuiltError(ChangegateError):
    """A command or check exists in the interface but is not built in this version.

    Does not establish: anything; no measurement was made.
    """
