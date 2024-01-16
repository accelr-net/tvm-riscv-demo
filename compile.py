import sys
import argparse

from inference.utils.helpers import pretty_print

from compiler.ModelBuilder import ModelBuilder


def main(args: argparse.Namespace) -> None:
  
  print("\n")
  print("-----------------------------------------------------")
  print("~ ACCELR RISC V TVM Demo - Model Compilation Script ~")
  print("-----------------------------------------------------")
  print("\n")

  x86ModelBuilder = ModelBuilder("x86_64")
  riscv64ModelBuilder = ModelBuilder("riscv64")

  if args.x86_64 or args.both_archs:
    pretty_print(" compiling for x86_64 ")
    if args.imagenet or args.all_models:
      print("   - imagenet... \n")
      x86ModelBuilder.build("imagenet")
    if args.kws or args.all_models:
      print("\n   - kws... \n")
      x86ModelBuilder.build("kws")
    pretty_print(" compilation for x86_64 completed ")

  if args.riscv64 or args.both_archs:
    pretty_print(" compiling for riscv64 ")
    if args.imagenet or args.all_models:
      print("   - imagenet... \n")
      riscv64ModelBuilder.build("imagenet")
    if args.kws or args.all_models:
      print("\n   - kws... \n")
      riscv64ModelBuilder.build("kws")
    pretty_print(" compilation for riscv64 completed ")

  print()
  print("---------------------")
  print("~ End of the Script ~")
  print("---------------------")
  print("\n")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("-i", "--imagenet",   help="activate imagenet model",          default=False)
  parser.add_argument("-k", "--kws",        help="activate kws model",               default=False)
  parser.add_argument("-a", "--all_models", help="activate all models",              default=False)

  parser.add_argument("-x", "--x86_64",     help="compile for x86_64 architecture",  default=False)
  parser.add_argument("-r", "--riscv64",    help="compile for riscv64 architecture", default=False)
  parser.add_argument("-b", "--both_archs", help="compile for both architectures",   default=False)

  args = parser.parse_args()

  if len(sys.argv) == 1:
    parser.print_help()
    print("\n")
    sys.exit(1)

  main(args)
