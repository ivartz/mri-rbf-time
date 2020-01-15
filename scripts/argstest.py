import argparse
# Define command line options
# this also generates --help and error handling
CLI=argparse.ArgumentParser()
CLI.add_argument(
  "--nifti",  # name on the CLI - drop the `--` for positional/required parameters
  nargs="*",  # 0 or more values expected => creates a list
  help="Takes in the relative file path + \
  file name of two or more co-registrated NIFTI1 files, \
  separated by space",
  type=str,
  default=["1.nii", "2.nii"],
)
CLI.add_argument(
  "--timeint",
  nargs="*",
  help="The time interval between the NIFTI1 file scans, in days, separated by space",
  type=int,
  default=[7],
)

# Parse the command line
args = CLI.parse_args()


# access CLI options
print(args.nifti)
print(args.timeint[0]+1)

#print("nifti: %r" % args.nifti)
#print("timeint: %r" % args.timeint)