import json

class Config(dict):
    def __init__(self, *args, **kw):
        super(Config, self).__init__(*args, **kw)

    def save(self, path):
        # write to disk
        with open(path, 'w') as outfile:
            json.dump(self, outfile, sort_keys = True, indent = 4)

    @staticmethod
    def load(path):
        with open(path, 'r') as infile:
            return Config(json.load(infile))

    def toString(self):

        str = json.dumps(self, indent=4, sort_keys=True) + '\r\n'
        if self['debug']:
            str+='\r\n' 
            str+='=========================DEBUGING========================='
            str+='\r\n'
        return str

    def print(self):
        print(self.toString())