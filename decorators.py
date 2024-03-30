import io
import os
import sys


def silent(func):
    def inner(*args):
        # Redirect standard output to null to avoid printing broken test results each time
        old_stdout = sys.stdout
        null = io.open(os.devnull, 'w')
        sys.stdout = null

        # Get function return
        res = func(*args)

        # Put standard output back online
        sys.stdout = old_stdout
        null.close()
        return res

    return inner
