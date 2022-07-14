from typing import Generic, Union, Iterable, Tuple, TypeVar

T_co = TypeVar('T_co', covariant=True)
I = TypeVar('I', int, str)

class Atts(object):
    def __init__(self, dict:dict=None, **kwargs) -> None:
        self.set(dict, kwargs)

    def __state__(self) -> dict:
        return vars(self)

    def set(self, dict:dict=None, **kwargs):
        if dict is not None:
            self.__dict__.update(dict)
        self.__dict__.update(kwargs)

    def reset(self) -> None:
        self.__dict__.clear()

class NamedDict(Generic[T_co]):
    def __init__(
            self,
            data:Union[Iterable[T_co], T_co] = None,
            meta:Union[Iterable[I],Iterable[Tuple[int, str]]]=None) -> None:
        self.data = {}
        self.name = {}
        self.id = {}
        self.length = 0
        self.insert(data, meta)

    def insert(
            self,
            data:Union[Iterable[T_co], T_co] = None,
            meta:Union[I, Iterable[I],Iterable[Tuple[int, str]]]=None
        ) -> None:
        if meta is not None: #Name(id) is passed
            if isinstance(data, T_co): #Single
                if isinstance(meta, int):
                    if meta in self.data:
                        self.ch_id(meta, self.length)
                    self.data[meta] = data
                elif isinstance(meta, str):
                    self.name[self.length] = meta
                    self.id[meta] = self.length
                    self.data[self.length] = data
                elif isinstance(meta[0], tuple):
                    if meta[0][0] in self.data:
                        self.ch_id(meta[0][0], self.length)
                    self.name[meta[0][0]] = meta[0][1]
                    self.id[meta[0][1]] = meta[0][0]
                    self.data[meta[0][0]] = self.data[meta[0][1]] = data
                elif isinstance(meta[0], str):
                    self.name[self.length] = meta[0]
                    self.id[meta[0]] = self.length
                    self.data[self.length] = self.data[meta[0]] = data
                else:
                    self.data[meta[0]] = data
                self.length += 1
            else: #list
                if isinstance(meta, int) or isinstance(meta, str):
                    raise TypeError("data and meta must have same number of elements.")
                if isinstance(meta[0], tuple):
                    for t in meta:
                        if t[1] == "":
                            continue
                        if t[0] in self.data:
                            self.ch_id(t[0], self.length)
                        self.name[t[0]] = t[1]
                        self.id[t[1]] = t[0]
                        self.data[t[0]] = self.data[t[1]] = data
                        self.length += 1
                elif isinstance(meta[0], str):
                    for s in meta:
                        if s == "":
                            continue
                        self.name[self.length] = s
                        self.id[s] = self.length
                        self.data[self.length] = self.data[s] = data
                        self.length += 1
                else:
                    for t in meta:
                        if t in self.data:
                            self.ch_id(t, self.length)
                        self.data[t] = data
                        self.length += 1

    def pop(self, index:I) -> T_co:
        if isinstance(index, int):
            name = self.name.pop(index, None)
            self.id.pop(name, None)
            self.data.pop(name, None)
            return self.data.pop(index, None)        
        elif isinstance(index, str):
            id = self.id.pop(index, None)
            self.name.pop(id, None)
            self.data.pop(id, None)
            return self.data.pop(index, None)
            
    def drop(self, index:Union[I, Iterable[I]]) -> None:
        if isinstance(index, int) or isinstance(index, str):
            self.pop(index)
        else:
            for i in index:
                self.pop(index)

    def clear(self) -> None:
        self.data = {}

    def sub(self, index:Union[Tuple[int, int], Iterable[int], Iterable[str]]):
        ret = NamedDict()
        if isinstance(index, tuple):
            for i in range(index[0], index[1]):
                ret.insert(self.data[i], ["" if not i in self.name else self.name[i]])
        else:
            if isinstance(index[0], str):
                for s in index:
                    ret.insert(self.data[s])
            else:
                for i in index:
                    ret.insert(self.data[i])
        return ret

    def ch_id(self, id:int, to:int) -> None:
        self.data[to] = self.data.pop(id)
        self.name[to] = self.name.pop(id)

    def __getitem__(self, id:I) -> T_co:
        return self.data[id]

    def __setitem__(self, id:I, value:T_co) -> None:
        if isinstance(id, int):
            id2 = self.name[id]
        else:
            id2 = self.id[id]
        if id2 is not None:
            self.data[id2] = value
        self.data[id] = value

    def __len__(self) -> int:
        return self.data.__len__()

    def __str__(self) -> str:
        return '\n'.join(self.data)