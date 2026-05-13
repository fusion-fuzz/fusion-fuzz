import os


bugs = os.listdir("./")


for each in bugs:
    if "." in each:
        continue
    outpath = f"./{each}/report.md"
    if not os.path.exists(outpath):
        continue
    f = open(outpath, 'r')
    out = f.read()
    f.close()

    if "AssertionError: {" in out or "<class 'TimeoutError'>" in out or "_interpchannelsmodule.c:2996" in out or "Modules/_io/iobase.c:949" in out or "tuple_new(cls, iterable)" in out or "encoder_listencode_dict ../Modules/_json.c:1855" in out or "SystemError: <method 'close' of '_io.BufferedWriter' objects> returned a result with an exception set" in out or "PyFunction_NewWithQualName ../Objects/funcobject.c:159" in out or "create_localdummies ../Modules/_threadmodule.c:1617" in out or "alloc_threadstate ../Python/pystate.c:1397" in out or "_asynciomodule.c:3147" in out or "iobase.c:956" in out or "../Objects/call.c:120: _PyObject_VectorcallDictTstate: Assertion" in out or "../Modules/_io/bytesio.c:616: _io_BytesIO_readinto_impl: Assertion" in out or "../Objects/rangeobject.c:243: compute_range_length: Assertion" in out or "../Modules/_io/textio.c:2848: _io_TextIOWrapper_tell_impl: Assertion" in out or "enc('spam" in out or "update_indent_cache: Assertion" in out or "../Objects/call.c:120: PyObject *_PyObject_VectorcallDictTstate" in out or "_io/bytesio.c:616: PyObject *_io_BytesIO_readinto_impl" in out or "Direct leak of 72000 byte(s) in 1 object" in out or "Direct leak of 1104 byte" in out or "40 byte(s) leaked" in out or "optimizer.c:144: int _Py" in out or "optimizer.c:1534: void unl" in out or "_json.c:1388: int upd" in out or "_PyJit_TryInitializeTracing /home/fuzz/WorkSpace/flowfusion-cpython/cpython/build/../Python/optimizer.c:981:70" in out or "Indirect leak of" in out or "AddressSanitizer: BUS on unknown address" in out or "_POP_TOP_INT.c:119: _Py_CODEUNIT" in out or "_PyUnicode_DecodeUnicodeEscapeInternal2: Assert" in out or "_PyEval_EvalFrameDefault(PyThreadState *, _PyInterpreterFrame *, int): Ass" in out:
        continue
    if "SUMMARY:" in out or "SystemError:" in out or "(core dumped)" in out or "Assertion failed:" in out or ": Assertion `" in out:
        codepath = f"./{each}/reproduce.py"
        f = open(codepath, 'r')
        code = f.read()
        f.close()
        if "ctypes" in code or "_testcapi" in code or "_testinternalcapi" in code or "_testlimit" in code or "AssemblerTestCase" in code or "array.array(" in code or "NUM_THREADS = 4" in code or "import ast" in code or "requires('curses')" in code or ".__code__ =" in code or "r = gc.get_referrers(thingy)" in code or "co_kwonlyargcount" in code or "functools.partial(extract_sig)" in code or "tokenize._generate_tokens_from_c_tokenizer" in code or "self.closed = NotImplemented" in code or "copy.deepcopy(param)" in code or ".fork()" in code or ("addaudithook" in code and "hashlib" in code) or "tuple_new(cls, iterable)" in code or "raise SystemError(" in code or "os.kill(os.getpid(), signal.SIGSEGV)" in code or "out, _ = p.communicate(" in code or "call_state_registration_func" in code or "multiprocessing.shared_memory.ShareableList" in code or "faulthandler.dump_traceback_later(" in code or "threading.stack_size(" in code or "threading.settrace" in code or "__code__" in code or "fcntl.fcntl(" in code:
            continue


        if "_ssl__SSLSocket_write_impl" in out:
            continue

        if ": Assertion `" in out:
            if "python: ../Python/importdl.c" in out or "python: ../Objects/codeobject.c:" in out:
                continue


        if "leaked" in out:
            if "atexit" in code or "curse" in code or "script_helper" in code or "make_executor_from_uops" in out or "fork()" in code or "pack(*args)" in code or "ConstructsNone" in code or "from xml.parsers import expat" in code:
                continue

        print(each)