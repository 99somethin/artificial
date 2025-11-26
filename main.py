import argparse
from lab1_face_eye import run_lab1
from lab2_features import run_lab2
from lab3_contours import run_lab3

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="lab", required=True)
    p1 = sub.add_parser("lab1"); p1.add_argument("--camera", type=int, default=0)
    p2 = sub.add_parser("lab2"); p2.add_argument("--template", required=True); p2.add_argument("--camera", type=int, default=0)
    p3 = sub.add_parser("lab3"); p3.add_argument("--templates_dir", required=True); p3.add_argument("--camera", type=int, default=0)
    args = p.parse_args()
    if args.lab == "lab1":
        run_lab1(args.camera)
    elif args.lab == "lab2":
        run_lab2(args.template, args.camera)
    elif args.lab == "lab3":
        run_lab3(args.templates_dir, args.camera)

if __name__ == "__main__":
    main()