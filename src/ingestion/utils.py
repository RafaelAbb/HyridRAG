import os
from datetime import datetime


def get_mod_time(file_path: str) -> datetime:
    '''Returns the last modification time of a file as a datetime object.'''
    timestamp = os.path.getmtime(file_path)
    return datetime.fromtimestamp(timestamp)
