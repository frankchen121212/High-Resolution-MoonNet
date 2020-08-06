"""
Usage: test <json_file>
"""

import docopt,json,os
from lib.core import test
from lib.dataset import DeepMoon
from lib.model import Unet,HRnet


if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    args = json.load(open(args["<json_file>"]))
    Data, Craters = None, None
    model = None

    # dataset
    if args["dataset"] == "deepmoon":
        Data, Craters = DeepMoon(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args["gpu"]
    # model
    if args["model"] == 'u-net':
        model = Unet(dim=args["input_length"],
                     learn_rate=args["lr"],
                     lmbda=args["lambda"],
                     drop=args["dp_rate"],
                     FL=args["filter_length"],
                     init=args["init"],
                     n_filters=args["n_filters"])
    elif args["model"] == 'high-resolution-net':
        model = HRnet(epochs=args["epochs"],
                      dim=args["input_length"],
                      learn_rate=args["lr"])
    test(args, Data, Craters, model)