"""Small shared regex helpers."""
import re

# digit run, allowing "1,234"/"3.5" as one token without swallowing trailing punctuation
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def compile_word_boundary_pattern(keywords):
    """One compiled regex matching any of `keywords` as a whole word,
    case-insensitively -- longest keywords first, so e.g. "smith machine"
    is tried before the shorter "smith" would otherwise win instead."""
    alternation = "|".join(re.escape(keyword) for keyword in sorted(keywords, key=len, reverse=True))
    return re.compile(rf"\b({alternation})\b", re.IGNORECASE)


def format_range(low, high):
    """{low, high} numbers -> "3" / "8-12" / "5.5"; high=None means a single value, not open-ended."""
    def fmt(number):
        return str(int(number)) if number == int(number) else str(number)
    return fmt(low) if high is None or high == low else f"{fmt(low)}-{fmt(high)}"


def _demo():
    assert format_range(8.0, 12.0) == "8-12"
    assert format_range(2.0, None) == "2"
    assert format_range(5.5, 5.5) == "5.5"
    assert NUMBER.findall("3x8-12, 90s") == ["3", "8", "12", "90"]

    pattern = compile_word_boundary_pattern(["smith", "smith machine"])
    assert pattern.search("Smith Machine Squat").group(0).lower() == "smith machine"
    assert pattern.search("Cable crossover") is None, "must not match unrelated substrings"
    assert pattern.search("flat bench") is None, "\\b must not match 'lat' inside 'flat'"


if __name__ == "__main__":
    _demo()
    print("ok")
