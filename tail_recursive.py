from functools import wraps

class TailRecursive:
    CONTINUE = object()
    is_first_call = [ True ]
    func = [ None ]
    arg  = [ None, None ]

    def __init__(self):
        pass

    def __call__(self, func):
        self.this_func = func

        @wraps(func)
        def _func(*args, **kwargs):
            self.func[0] = self.this_func
            self.arg[0]  = args
            self.arg[1]  = kwargs

            if self.is_first_call[0]:
                self.is_first_call[0] = False

                result = self.CONTINUE
                try:
                    while result is self.CONTINUE:
                        result = self.func[0](*self.arg[0], **self.arg[1])
                finally:
                    self.is_first_call[0] = True
                    self.arg[0] = None
                    self.arg[1] = None

                return result

            else:
                return self.CONTINUE

        return _func
