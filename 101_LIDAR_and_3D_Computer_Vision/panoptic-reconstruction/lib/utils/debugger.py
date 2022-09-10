import socket

import argparse


class Debugger:
    def __init__(self, local_ip="127.0.0.1", port=12345, timeout=10):
        if timeout is not None:
            socket.setdefaulttimeout(timeout)

        import pydevd_pycharm
        try:
            print(f"Trying to connect to debugger within {timeout}s")
            pydevd_pycharm.settrace(local_ip, port=port, stdoutToServer=True, stderrToServer=True,
                                    patch_multiprocessing=False, suspend=False)
        except socket.timeout:
            print(f"No debugger found within {timeout}s")


class ArgParserWithDebugger:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-d", "--debug", action="store_true")

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def parse_args(self, args=None):
        args = self.parser.parse_args(args)
        if args.debug:
            Debugger()

        return args