'''Utility functions used in other scripts'''

from pathlib import Path
from typing import Optional
import re


root_dir = Path(__file__).parent.parent


def strip_url(url: Optional[str]) -> str:
    '''Remove the GET request parameters from a url.
    Args:
        url (str or None): Input url.
    Returns:
        str: `url` without GET request parameters, or None if `url` is None.
    '''
    if url is None:
        return None
    url = re.sub(r'\?.*', '', url)
    url = re.sub(r'\/$', '', url)
    return url
