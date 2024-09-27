import subprocess
import os
import argparse

# Conda packages
conda_packages = [
    "_libgcc_mutex=0.1=main",
    "_openmp_mutex=5.1=1_gnu",
    "asttokens=2.4.1=pyhd8ed1ab_0",
    "blas=1.0=mkl",
    "bzip2=1.0.8=h5eee18b_5",
    "ca-certificates=2024.7.4=hbcca054_0",
    "certifi=2024.7.4=pyhd8ed1ab_0",
    "charset-normalizer=2.0.4=pyhd3eb1b0_0",
    "comm=0.2.2=pyhd8ed1ab_0",
    "cuda-cudart=11.8.89=0",
    "cuda-cupti=11.8.87=0",
    "cuda-libraries=11.8.0=0",
    "cuda-nvrtc=11.8.89=0",
    "cuda-nvtx=11.8.86=0",
    "cuda-runtime=11.8.0=0",
    "ffmpeg=4.3=hf484d3e_0",
    "freetype=2.12.1=h4a9f257_0",
    "gmp=6.2.1=h295c915_3",
    "gnutls=3.6.15=he1e5248_0",
    "intel-openmp=2023.1.0=hdb19cb5_46306",
    "ipykernel=6.29.5=pyh3099207_0",
    "ipython=8.26.0=pyh707e725_0",
    "jpeg=9e=h5eee18b_1",
    "lame=3.100=h7b6447c_0",
    "lcms2=2.12=h3be6417_0",
    "ld_impl_linux-64=2.38=h1181459_1",
    "lerc=3.0=h295c915_0",
    "libcublas=11.11.3.6=0",
    "libcufft=10.9.0.58=0",
    "libcufile=1.9.0.20=0",
    "libcurand=10.3.5.119=0",
    "libcusolver=11.4.1.48=0",
    "libcusparse=11.7.5.86=0",
    "libdeflate=1.17=h5eee18b_1",
    "libffi=3.4.4=h6a678d5_0",
    "libgcc-ng=11.2.0=h1234567_1",
    "libgomp=11.2.0=h1234567_1",
    "libiconv=1.16=h7f8727e_2",
    "libidn2=2.3.4=h5eee18b_0",
    "libjpeg-turbo=2.0.0=h9bf148f_0",
    "libnpp=11.8.0.86=0",
    "libnvjpeg=11.9.0.86=0",
    "libpng=1.6.39=h5eee18b_0",
    "libsodium=1.0.18=h36c2ea0_1",
    "libstdcxx-ng=11.2.0=h1234567_1",
    "libtasn1=4.19.0=h5eee18b_0",
    "libtiff=4.5.1=h6a678d5_0",
    "libunistring=0.9.10=h27cfd23_0",
    "libuuid=1.41.5=h5eee18b_0",
    "libwebp-base=1.3.2=h5eee18b_0",
    "llvm-openmp=14.0.6=h9e868ea_0",
    "lz4-c=1.9.4=h6a678d5_0",
    "mkl=2023.1.0=h213fc3f_46344",
    "mkl_fft=1.3.8=py311h5eee18b_0",
    "mkl_random=1.2.4=py311hdb19cb5_0",
    "mpc=1.1.0=h10f8cd9_1",
    "mpfr=4.0.2=hb69a4c5_1",
    "ncurses=6.4=h6a678d5_0",
    "openh264=2.1.1=h4ff587b_0",
    "openjpeg=2.4.0=h3ad879b_0",
    "openssl=3.0.14=h5eee18b_0",
    "python=3.11.8=h955ad1f_0",
    "python_abi=3.11=2_cp311",
    "pytorch=2.2.1=py3.11_cuda11.8_cudnn8.7.0_0",
    "pytorch-cuda=11.8=h7e8668a_5",
    "pytorch-mutex=1.0=cuda",
    "readline=8.2=h5eee18b_0",
    "sqlite=3.41.2=h5eee18b_0",
    "tbb=2021.8.0=hdb19cb5_0",
    "tk=8.6.12=h1ccaba5_0",
    "typing_extensions=4.9.0=py311h06a4308_1",
    "zeromq=4.3.5=h6a678d5_0",
    "zlib=1.2.13=h5eee18b_0",
    "zstd=1.5.5=hc292b87_0"
]

# Pip packages
pip_packages = [
    "av==12.0.0",
    "blessed==1.20.0",
    "contourpy==1.2.0",
    "cycler==0.12.1",
    "debugpy==1.6.7",
    "decorator==5.1.1",
    "exceptiongroup==1.2.2",
    "executing==2.0.1",
    "filelock==3.13.1",
    "fonttools==4.50.0",
    "fsspec==2024.3.1",
    "gmpy2==2.1.2",
    "gpustat==1.1.1",
    "h5==0.9.3",
    "h5py==3.11.0",
    "huggingface-hub==0.24.6",
    "idna==3.4",
    "importlib-metadata==8.2.0",
    "jedi==0.19.1",
    "jinja2==3.1.3",
    "joblib==1.4.2",
    "jupyter-core==5.7.2",
    "kiwisolver==1.4.5",
    "markupsafe==2.1.3",
    "matplotlib==3.8.3",
    "mkl-fft==1.3.8",
    "mkl-random==1.2.4",
    "mkl-service==2.4.0",
    "mpmath==1.3.0",
    "nest-asyncio==1.6.0",
    "networkx==3.1",
    "numpy==1.26.4",
    "nvidia-ml-py==12.535.133",
    "opencv-python==4.9.0.80",
    "packaging==24.0",
    "pandas==2.2.1",
    "parso==0.8.4",
    "pexpect==4.9.0",
    "pickleshare==0.7.5",
    "pillow==10.2.0",
    "pip==23.3.1",
    "platformdirs==4.2.2",
    "prompt-toolkit==3.0.47",
    "psutil==5.9.0",
    "pure_eval==0.2.3",
    "pygments==2.18.0",
    "pyparsing==3.1.2",
    "python-dateutil==2.9.0.post0",
    "python-graphviz==0.20.3",
    "pytz==2024.1",
    "pyyaml==6.0.1",
    "pyzmq==25.1.2",
    "regex==2023.12.25",
    "requests==2.31.0",
    "scikit-learn==1.3.2",
    "scipy==1.11.3",
    "setuptools==68.1.2",
    "six==1.16.0",
    "sqlite==3.41.2",
    "sympy==1.13",
    "tbb==2021.8.0",
    "torch==2.2.1",
    "torchvision==0.16.2",
    "traitlets==5.9.0",
    "tqdm==4.66.1",
    "typing-extensions==4.9.0",
    "urllib3==1.27.0",
    "wcwidth==0.2.10",
    "wheel==0.41.2",
    "wrapt==1.15.0",
    "zipp==3.17.1"
]

# Function to create a conda environment with a specific location
def create_conda_env(env_path, python_version):
    print(f"Creating conda environment at '{env_path}' with Python {python_version}...")
    subprocess.run(["conda", "create", "--prefix", env_path, f"python={python_version}", "-y"])

# Function to install Conda packages in the environment
def install_conda_packages(env_path, packages):
    for package in packages:
        print(f"Installing {package} via conda in environment '{env_path}'...")
        subprocess.run(["conda", "install", "--prefix", env_path, "-y", package])

# Function to install Pip packages in the environment
def install_pip_packages(env_path, packages):
    for package in packages:
        print(f"Installing {package} via pip in environment '{env_path}'...")
        subprocess.run(["conda", "run", "--prefix", env_path, "pip", "install", package])

# Main function to handle the creation of the environment and installation
def setup_env(env_path, python_version, conda_packages, pip_packages):
    # Create the conda environment
    create_conda_env(env_path, python_version)

    # Install conda packages
    install_conda_packages(env_path, conda_packages)

    # Install pip packages
    install_pip_packages(env_path, pip_packages)

# Argument parser for command-line input
def parse_arguments():
    parser = argparse.ArgumentParser(description="Create a conda environment with a custom installation location.")
    parser.add_argument("env_name", type=str, help="Name of the Conda environment")
    parser.add_argument("install_location", type=str, help="Path to the installation location")
    parser.add_argument("--python_version", type=str, default="3.11.8", help="Python version to use in the environment")
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    
    # Construct the full environment path
    env_path = os.path.join(args.install_location, args.env_name)
    
    # Set up the environment with the provided inputs
    setup_env(env_path, args.python_version, conda_packages, pip_packages)
