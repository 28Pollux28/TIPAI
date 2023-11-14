import argparse
import os
import sys


def get_args():
    """
    Get the arguments from the command line
    --action (required) : Train or test
    If train:
        --model (required) : Model name
        --data (required) : Dataset folder
        --csv (required) : CSV file
        --batch-size (optional): Batch size (default 16)
        --epochs (optional): Number of epochs (default 20)
        --lr (optional): Learning rate (default 0.001)
        --wd (optional): Weight decay (default 0.0001)
        --fine-tune (optional): Fine tuning (default False)
        --run (optional) : run number
    If test:
        --model-file (required) : Model file
        --data (required) : Dataset folder
        --out-csv (required) : Out CSV file
        --batch-size (optional) : Batch size (default 16)
    :return: Parsed arguments from the command line in a list of strings
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", required=True,
                        help="Train or test")
    parser.add_argument("--model", required=False,
                        help="Model name")
    parser.add_argument("--data", required=False,
                        help="Dataset folder")
    parser.add_argument("--csv", required=False,
                        help="CSV file")
    parser.add_argument("--model-file", required=False,
                        help="Model file")
    parser.add_argument("--out-csv", required=False,
                        help="Out CSV file")
    parser.add_argument("--batch-size", required=False,
                        help="Batch size", default=16, type=int)
    parser.add_argument("--epochs", required=False,
                        help="Number of epochs", default=20, type=int)
    parser.add_argument("--lr", required=False,
                        help="Learning rate", default=0.001, type=float)
    parser.add_argument("--wd", required=False,
                        help="Weight decay", default=0.0001, type=float)
    parser.add_argument("--fine-tune", required=False,
                        help="Fine tuning", default=False, type=bool)
    parser.add_argument("--run", required=False,
                        help="Run number", default=None, type=int)
    parser.add_argument("--class-number", required=False,
                        help="Class number", default=2, type=int)
    parser.add_argument("--use-scheduler", required=False,
                        help="Use scheduler", default=False, type=bool)
    args = parser.parse_args()
    if args.action == "train":
        if args.model is None:
            print("Argument --model is required for training")
            sys.exit(1)
        if args.data is None:
            print("Argument --data is required for training")
            sys.exit(1)
        if args.csv is None:
            print("Argument --csv is required for training")
            sys.exit(1)
    elif args.action == "test":
        if args.model_file is None:
            print("Argument --model-file is required for testing")
            sys.exit(1)
        if args.data is None:
            print("Argument --data is required for testing")
            sys.exit(1)
        if args.out_csv is None:
            print("Argument --out-csv is required for testing")
            sys.exit(1)
        if args.class_number:
            args.class_number = int(args.class_number)
    else:
        print("Argument -1 or --action must be train or test")
        sys.exit(1)
    return args


def init_run(run_number):
    if run_number is None:
        # If no run number is specified, find the last run number and increment it
        # If no run exists, start from 0
        if os.path.exists('runs'):
            runs = [int(f.split('-')[2]) for f in os.listdir('runs') if f.startswith('ISIIC-experiement-')]
            if len(runs) == 0:
                run_number = 0
            else:
                run_number = max(runs) + 1
        else:
            run_number = 0
    if not os.path.exists('runs/ISIIC-experiement-'+str(run_number)):
        os.makedirs('runs/ISIIC-experiement-'+str(run_number))
        os.makedirs('runs/ISIIC-experiement-'+str(run_number)+'/models')
    return run_number

