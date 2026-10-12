import argparse
import shutil
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compiler', default='g++')
    parser.add_argument('--build-dir', default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    build = Path(args.build_dir).resolve() if args.build_dir else root / 'lab/lab2/build'
    build.mkdir(parents=True, exist_ok=True)
    compiler = shutil.which(args.compiler) or args.compiler
    suffix = '.exe' if __import__('os').name == 'nt' else ''
    for name, source in [('lab2_gen','gen.cpp'),('lab2_logistic','main.cpp'),('lab2_nb','nb.cpp'),('lab2_verify','verify.cpp')]:
        command = [compiler,'-std=c++20','-O2','-Wall','-Wextra','-pedantic','-I',str(root/'include'),'-I',str(root),str(root/'lab/lab2'/source),'-o',str(build/(name+suffix))]
        subprocess.run(command,check=True)
        print('built:',build/(name+suffix),flush=True)
    subprocess.run([str(build/('lab2_verify'+suffix))],check=True)

if __name__ == '__main__':
    main()
