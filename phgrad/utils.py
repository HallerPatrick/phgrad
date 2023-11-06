import inspect
from typing import get_type_hints
from types import MethodType
from functools import partialmethod


def register(name, fxn, cls):

    partial_method = partialmethod(fxn.apply, fxn)

    # TODO: This is not working
    # def wrapper(self, *args, **kwargs):
    #     method = partial_method.__get__(self)
    #     return method(*args, **kwargs)
    
    # forward_signature = inspect.signature(fxn.forward)
    # return_annotation = forward_signature.return_annotation

    # new_params = tuple(list(forward_signature.parameters.values()))
    # new_signature = inspect.Signature(parameters=new_params, return_annotation=return_annotation)

    # wrapper.__signature__ = new_signature
    # wrapper.__doc__ = fxn.__doc__
    
    # method = MethodType(wrapper, cls)
    setattr(cls, name, partial_method)



def generate_stub_for_class(cls, filename):
    class_name = cls.__name__
    stub_lines = ["from typing import *", "from numpy import ndarray", f"class {class_name}:"]
    
    # Get all methods and properties of the class
    for name, value in inspect.getmembers(cls):
        # print(name, value)
        if inspect.isfunction(value) or inspect.ismethod(value):
            try:
                isstaticmethod = isinstance(inspect.getattr_static(cls, value.__name__), staticmethod)
            except AttributeError:
                isstaticmethod = False
            try:
                isclassmethod = isinstance(inspect.getattr_static(cls, value.__name__), classmethod)
            except AttributeError:
                isclassmethod = False

            try:
                isproperty = isinstance(inspect.getattr_static(cls, value.__name__), property)
            except AttributeError:
                isproperty = False

            # Get return type
            return_annotation_repr = inspect.signature(value).return_annotation

            if "numpy.ndarray" in str(return_annotation_repr):
                return_annotation = "Tensor"
            elif return_annotation_repr == inspect.Signature.empty:
                return_annotation = None
            else:
                return_annotation = return_annotation_repr

            # Get type hints for the method
            hints = get_type_hints(value)
            args = []

            if not isstaticmethod:
                args.append('self')

            for arg in inspect.signature(value).parameters.values():
                arg_name = arg.name
                if arg_name == 'self':
                    continue
                elif arg_name in hints:
                    args.append(f"{arg_name}: {hints[arg_name]}")
                elif arg_name == 'kwargs':
                    args.append("**kwargs")
                elif arg_name == 'args':
                    args.append("*args")
                elif arg_name == "tensor":
                    args.append(f"{arg_name}: Tensor")
                else:
                    args.append(arg_name)

                if arg.default != inspect.Parameter.empty:
                    args[-1] += f"={arg.default}"

            if len(args) == 1:
                args = args[0]
            else:
                args = ", ".join(args)

            if return_annotation:
                stub_lines.append(f"    def {name}({args}) -> {return_annotation}: ...")
            else:
                stub_lines.append(f"    def {name}({args}): ...")

        elif not name.startswith('__'):
            # Assume attributes are of type Any for the stub
            stub_lines.append(f"    {name}: Any")

    # Join the lines and write to a .pyi file
    stub_content = "\n".join(stub_lines)
    stub_filename = f"phgrad/{filename}.pyi"
    
    with open(stub_filename, 'w') as stub_file:
        stub_file.write(stub_content)