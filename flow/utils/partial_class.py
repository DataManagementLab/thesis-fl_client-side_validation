import functools

def partial_class(cls, *args, **kwds):

    class PartialClass(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwds)

    return PartialClass
