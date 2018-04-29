from functools import wraps

class TailRecursive:
    def __init__(self):
        self.CONTINUE = object()
        self.is_first_call = True
        self.func   = None
        self.args   = None
        self.kwargs = None

    def __call__(self, func):
        @wraps(func)
        def _func(*args, **kwargs):
            self.func   = func
            self.args   = args
            self.kwargs = kwargs

            if self.is_first_call:
                self.is_first_call = False

                result = self.CONTINUE
                try:
                    while result is self.CONTINUE:
                        result = self.func(*self.args, **self.kwargs)
                finally:
                    self.is_first_call = True

                return result

            else:
                return self.CONTINUE

        return _func
