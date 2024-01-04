import sys
import platform
import argparse

from inference.session import InferenceSession as inference_session

from inference.utils.helpers import pretty_print


def main(args: argparse.Namespace) -> None:

  print("\n")
  print("---------------------------------------------------")
  print("~ ACCELR RISC V TVM Demo - Model Inference Script ~")
  print("---------------------------------------------------")
  print("\n")

  architecture = platform.machine().lower()

  if (args.imagenet_pt or args.all_models) and architecture == "x86_64":
    pretty_print(f" Pytorch imagenet inference session on {architecture} ")
    pytorch_imagenet_session = inference_session("imagenet", architecture, args.numsteps, pt=True)
    pytorch_imagenet_session.run()
    pytorch_imagenet_session.benchmark()
    pretty_print(" end of Pytorch imagenet inference session ")

  if args.imagenet or args.all_models:
    pretty_print(f" TVM imagenet inference session on {architecture} ")
    imagenet_session = inference_session("imagenet", architecture, args.numsteps)
    imagenet_session.run()
    imagenet_session.benchmark()
    pretty_print(" end of TVM imagenet inference session ")

  if (args.kws_pt or args.all_models) and architecture == "x86_64":
    pretty_print(f" Pytorch kws inference session on {architecture} ")
    pytorch_kws_session = inference_session("kws", architecture, args.numsteps, pt=True)
    pytorch_kws_session.run()
    pytorch_kws_session.benchmark()
    pretty_print(" end of Pytorch kws inference session ")

  if args.kws or args.all_models:
    pretty_print(f" TVM kws inference session on {architecture} ")
    kws_session = inference_session("kws", architecture, args.numsteps)
    kws_session.run()
    kws_session.benchmark()
    pretty_print(" end of TVM kws inference session ")

  if architecture == "riscv64":
    print(" generating evaluation reports ... \n")
    if args.imagenet or args.all_models:
      imagenet_session.evaluate()
    if args.kws or args.all_models:
      kws_session.evaluate()

  print("\n")
  print("---------------------")
  print("~ End of the Script ~")
  print("---------------------")
  print("\n")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("-i", "--imagenet",    help="activate imagenet model",         default=False)
  parser.add_argument("-k", "--kws",         help="activate kws model",              default=False)
  parser.add_argument("-m", "--imagenet_pt", help="activate pytorch imagenet model", default=False)
  parser.add_argument("-w", "--kws_pt",      help="activate pytorch kws model",      default=False)
  parser.add_argument("-a", "--all_models",  help="activate all models",             default=False)
  parser.add_argument("-s", "--numsteps",    help="number of examples",              default=10)

  args = parser.parse_args()

  if len(sys.argv) == 1:
    parser.print_help()
    print("\n")
    sys.exit(1)

  main(args)
