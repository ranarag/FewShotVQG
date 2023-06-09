            
class Factory(object):
    def __init__(self):
        self._object_dict = {}
        
    def register_class(self, name, cl):
        self._object_dict[name] = cl
    
    def get_class_obj(self, name, *args, **kwargs):
#         try:
        obj = self._object_dict[name](*args, **kwargs)
#         except KeyError as e:
#             print("{} is not implemented or linked to factory class; exiting".format(name))
#             exit(1)
            
        return obj

    @property
    def list(self):
        for name, obj in self._object_dict.items():
            print("{} : {}".format(name, obj))
        