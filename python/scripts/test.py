"""
A simple test code for running C++ code in python environment
"""
from ctypes import *
from ctypes import POINTER

CLIB = "cmake-build-debug/libtest.dylib"

lib = cdll.LoadLibrary(CLIB)

lib.print_hello()

lib.print_message_from_python.argtypes = [c_char_p]
s = "Hello from Python"
lib.print_message_from_python(s.encode('utf-8'))