

# Proto Syntax
* required
must be initialized ,otherwise raise error,other than this, it is like an optional field

* optional
** default value:
numeric type: zero
strings : empty string
bool : false
embedded message: default instance or prototype which has none of its field set

* repeated
the field may be repeated any number of times(include zero), like dynamically sized arrays

## Note that using required is more harm than good



# api
CopyFrom
MergeFrom
IsInitialized
MergeFromString
ParseFromString
HasField
ClearField
WhichOneof

```
from google.protobuf import text_format
text_format.Merge("asdg",message)
```

