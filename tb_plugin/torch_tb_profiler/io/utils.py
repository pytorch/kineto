def as_str_any(value):
    """Converts to `str` as `str(value)`, but use `as_str` for `bytes`.

    Args:
    value: A object that can be converted to `str`.

    Returns:
    A `str` object.
    """
    if isinstance(value, bytes):
        return as_str(value)
    else:
        return str(value)


def as_text(bytes_or_text, encoding="utf-8"):
    """Returns the given argument as a unicode string.

    Args:
    bytes_or_text: A `bytes`, `str`, or `unicode` object.
    encoding: A string indicating the charset for decoding unicode.

    Returns:
    A `str` (Python 3) object.

    Raises:
    TypeError: If `bytes_or_text` is not a binary or unicode string.
    """
    if isinstance(bytes_or_text, str):
        return bytes_or_text
    elif isinstance(bytes_or_text, bytes):
        return bytes_or_text.decode(encoding)
    else:
        raise TypeError(
            "Expected binary or unicode string, got %r" % bytes_or_text
        )


# Convert an object to a `str` in both Python 2 and 3.
as_str = as_text


def as_bytes(bytes_or_text, encoding="utf-8"):
    """Converts either bytes or unicode to `bytes`, using utf-8 encoding for
    text.

    Args:
    bytes_or_text: A `bytes`, `str`, or `unicode` object.
    encoding: A string indicating the charset for encoding unicode.

    Returns:
    A `bytes` object.

    Raises:
    TypeError: If `bytes_or_text` is not a binary or unicode string.
    """
    if isinstance(bytes_or_text, str):
        return bytes_or_text.encode(encoding)
    elif isinstance(bytes_or_text, bytes):
        return bytes_or_text
    else:
        raise TypeError(
            "Expected binary or unicode string, got %r" % (bytes_or_text,)
        )


def parse_blob_url(url):
    from urllib import parse
    url_path = parse.urlparse(url)

    parts = url_path.path.lstrip('/').split('/', 1)
    return url_path.netloc, tuple(parts)
