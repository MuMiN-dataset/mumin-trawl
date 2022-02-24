'''Utility functions used in other scripts'''

from pathlib import Path
from typing import Optional
import re
import os
import shutil
import tempfile
from zipfile import ZipFile, ZIP_STORED, ZipInfo


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


class UpdateableZipFile(ZipFile):
    '''A zipfile object which can be updated.

    More or less a copy of https://stackoverflow.com/a/35435548/3154226.

    Usage:
    >>> with UpdateableZipFile("C:\Temp\Test2.docx", "a") as zip_file:
    ...
    ...     # Overwrite a file with a string
    ...     zip_file.writestr("word/document.xml", "Some data")
    ...
    ...     # exclude an exiting file from the zip
    ...     zip_file.remove_file("word/fontTable.xml")
    ...
    ...     # Write a new file (with no conflict) to the zp
    ...     zip_file.writestr("new_file", "more data")
    ...
    ...     # Overwrite a file with a file
    ...     zip_file.write(r"C:\Temp\example.png", "word/settings.xml")
    '''
    class DeleteMarker:
        pass

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Track file to override in zip
        self._replace = dict()

        # Whether the with statement was called
        self._allow_updates = False

    def writestr(self, zinfo_or_arcname, bytes, compress_type=None):

        if isinstance(zinfo_or_arcname, ZipInfo):
            name = zinfo_or_arcname.filename
        else:
            name = zinfo_or_arcname

        # If the file exists, and needs to be overridden, mark the entry, and
        # create a temp-file for it we allow this only if the with statement is
        # used
        if self._allow_updates and name in self.namelist():
            temp_file = self._replace.get(name, tempfile.TemporaryFile())
            self._replace[name] = temp_file
            temp_file.write(bytes)

        # Otherwise just act normally
        else:
            super().writestr(zinfo_or_arcname=zinfo_or_arcname,
                             bytes=bytes,
                             compress_type=compress_type)

    def write(self, filename, arcname=None, compress_type=None):
        arcname = arcname or filename

        # If the file exists, and needs to be overridden,
        # mark the entry, and create a temp-file for it
        # we allow this only if the with statement is used
        if self._allow_updates and arcname in self.namelist():
            temp_file = self._replace.get(arcname, tempfile.TemporaryFile())
            self._replace[arcname] = temp_file
            with open(filename, "rb") as source:
                shutil.copyfileobj(source, temp_file)

        # Otherwise just act normally
        else:
            super().write(filename=filename,
                          arcname=arcname,
                          compress_type=compress_type)

    def __enter__(self):
        self._allow_updates = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            super(UpdateableZipFile, self).__exit__(exc_type, exc_val, exc_tb)
            if len(self._replace) > 0:
                self._rebuild_zip()
        finally:
            # In case rebuild zip failed, be sure to still release all the temp
            # files
            self._close_all_temp_files()
            self._allow_updates = False

    def _close_all_temp_files(self):
        for temp_file in self._replace.values():
            if hasattr(temp_file, 'close'):
                temp_file.close()

    def remove_file(self, path):
        self._replace[path] = self.DeleteMarker()

    def _rebuild_zip(self):
        tempdir = tempfile.mkdtemp()
        try:
            temp_zip_path = Path(tempdir) / 'new.zip'
            with ZipFile(self.filename, 'r') as zip_read:

                # Create new zip with assigned properties
                with ZipFile(temp_zip_path,
                             mode='w',
                             compression=self.compression,
                             allowZip64=self._allowZip64) as zip_write:

                    for item in zip_read.infolist():

                        # Check if the file should be replaced / or deleted
                        replacement = self._replace.get(item.filename, None)

                        # If marked for deletion, do not copy file to new
                        # zipfile
                        if isinstance(replacement, self.DeleteMarker):
                            del self._replace[item.filename]
                            continue

                        # If marked for replacement, copy temp_file, instead of
                        # old file
                        elif replacement is not None:
                            del self._replace[item.filename]

                            # Write replacement to archive, and then close it
                            # (deleting the temp file)
                            replacement.seek(0)
                            data = replacement.read()
                            replacement.close()
                        else:
                            data = zip_read.read(item.filename)

                        zip_write.writestr(item, data)

            # Override the archive with the updated one
            shutil.move(temp_zip_path, self.filename)

        finally:
            shutil.rmtree(tempdir)
