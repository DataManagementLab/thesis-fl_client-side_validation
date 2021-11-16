import sys
from functools import partialmethod

def partialclass(cls, *args, **kwds):
    class PartialClass(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwds)

    return PartialClass

def partial_class(name, cls, *args, **kwds):
    new_cls = type(name, (cls,), {
        '__init__': partialmethod(cls.__init__, *args, **kwds)
    })

    # The following is copied nearly ad verbatim from `namedtuple's` source.
    """
    # For pickling to work, the __module__ variable needs to be set to the frame
    # where the named tuple is created.  Bypass this step in enviroments where
    # sys._getframe is not defined (Jython for example) or sys._getframe is not
    # defined for arguments greater than 0 (IronPython).
    """
    try:
        new_cls.__module__ = sys._getframe(1).f_globals.get('__name__', '__main__')
    except (AttributeError, ValueError):
        pass

    return new_cls