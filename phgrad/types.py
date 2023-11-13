from enum import Enum, auto


class _DType(Enum):
    """Our wrapper for backend data types"""
    bool = auto()
    float32 = auto() # float
    float64 = auto() # double
    int8 = auto() # byte
    uint8 = auto()
    int16 = auto()
    int32 = auto()
    int64 = auto()

class DType:

    def __init__(self, dtype: _DType):
        self._dtype = dtype

    def from_frontend(self, dtype: _DType):
        return DType(dtype)

    def __eq__(self, other):
        return self._dtype == other._dtype

    def __str__(self):
        return str(self._dtype)

    def __repr__(self):
        return repr(self._dtype)


bool = DType(_DType.bool)
float32 = DType(_DType.float32)
float64 = DType(_DType.float64)
int8 = DType(_DType.int8)
uint8 = DType(_DType.uint8)
int16 = DType(_DType.int16)
int32 = DType(_DType.int32)
int64 = DType(_DType.int64)
