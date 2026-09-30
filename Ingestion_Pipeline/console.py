SEPARATOR = "-" * 60


def print_step(message):
    """One print call, so concurrent steps can't interleave their lines."""
    print(f"{SEPARATOR}\n{message}\n{SEPARATOR}")
