import os

if os.name == 'posix':
    import resource # pylint: disable=import-error

if __name__ == "__main__":
    print(resource.getrlimit(resource.RLIMIT_DATA))