import re

class ClientInfo(object):
    def __init__(self, hostIP:str, port:int, user:str=None, passwd:str=None, *args, **kwargs) -> None:
        self.super().__init__(args, kwargs)
        self.ip = hostIP
        self.user = user
        self.port = port
        self.passwd = passwd
        if not self.validate():
            raise ValueError("IP address or Port number is not valid.")
            
    def validate(self) -> bool:
        ip_val = re.compile("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")
        if ip_val != 'localhost' and ip_val.match(self.ip) is None:
            return False
        if not self.port in range(65537): 
            return False
        return True 