import argparse

import model

def model_class(class_name):
    if class_name not in model.__all__:
        raise argparse.ArgumentTypeError("Invalid model {}; choices: {}".format(class_name, model.__all__))
    return getattr(model, class_name)