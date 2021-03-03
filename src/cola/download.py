# Johannes Siedersleben
# QAware GmbH, Munich
# 28.2.2021

# uncomment this if import fails
# !pip install wget

import os
import zipfile
import wget


def download(url, zipped_file, zipped_dir, unzipped_file) -> None:
    """
    download and unzip raw data
    :param url: URL to be downloaded from
    :param zipped_file: name of file to be downloaded
    :param zipped_dir: target directory
    :param unzipped_file: name of unzipped file
    :return: None
    """

    if not os.path.exists(zipped_file):
        wget.download(url, zipped_file)
    print('download successful')

    if not os.path.exists(zipped_dir):
        zip = zipfile.ZipFile(zipped_file)
        zip.extractall()

    print('unzipped file now at ' + unzipped_file)
