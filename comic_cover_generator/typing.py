"""Typing utility module."""
import platform

if float(platform.sys.version[:3]) < 3.8:
    from typing_extensions import Protocol, TypedDict
else:
    from typing import Protocol, TypedDict
