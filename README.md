<div align="center">
  <a href="https://accelr.lk/">
    <img src="https://avatars.githubusercontent.com/u/55974019?s=200&v=4" alt="Logo" width="80" height="80">
  </a>

<h1 align="center">TVM RISC-V Demo</h1>

<p align="center">
   Demonstrator on running TVM on RISC-V with ResNet18 and KWS examples
    <br />
  </p>
</div>
</p>

## Introduction
This repository contains the demo application we have created to run and benchmark Image Classification and KWS (Key Word Spotting) models on a [QEMU](https://www.qemu.org/) emulated [RV64IMFD](https://www3.diism.unisi.it/~giorgi/didattica/tools1/RISCV_ISA_TABLE-v11.pdf) Ubuntu Linux system through [Apache TVM](https://tvm.apache.org/).

## Demonstration Workflow
Fist follow the instructions [here](./docs/ubuntu_qemu.md)  prepare a QEMU emulated RISC-V environment on an x86_64 system. Additionally, create a shared directory between the host and guest according to given guidelines. *For the rest of the guide, x86_64 system will be called the "host", and RISC-V system will be called the "guest".*

---

### Clone & build Apache TVM on host

Clone the repo and sync submodules.
```bash
git clone https://github.com/apache/tvm.git
cd tvm
```

We use TVM version 0.10.0, checkout to the tag v0.10.0 and sync submodules.
```bash
git checkout v0.10.0
git submodule init
git submodule update
```

Execute the following commands to install the dependencies for TVM.
```bash
sudo apt-get update
sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
```

Create the build folder for TVM and copy the `config.cmake` file to the build folder.
```bash
mkdir build
cp cmake/config.cmake build
cd build
```

Edit the copied file as follows to configure the build.

Change expression `set(USE_LLVM OFF)` to `set(USE_LLVM ON)`.

__Note:__
The above change means the TVM includes the binaries from the LLVM installation in your computer for TVM code generation. Therefore It is required to have LLVM installed beforehand.

You can check whether you have an already installed version of LLVM in your computer by executing the following command.
```bash
llc --version
```

If not install LLVM by executing the following command.
```bash
bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"
```

A more comprehensive guide to LLVM installation from pre-built binaries can be found [here](https://apt.llvm.org/).

Configure and execute the build with the following commands (replace 4 with number of parallel compile jobs you need to run in your computer).
```bash
cmake ..
make -j4
```

Now run the python installation to register TVM as a pip package.
```bash
cd ../python
python setup.py install --user
cd ..
```

Note that the `--user` flag is not necessary if you’re installing to a managed local environment, like `virtualenv`.

TVM is installed in the host now. You can check the instillation by executing the following command.
```bash
python -c "import tvm; print(tvm.__version__)"
```

A comprehensive guide for the build process can be found [here](https://tvm.apache.org/docs/v0.10.0/install/from_source.html).

---

### Clone & build Apache TVM Runtime on guest

Clone the repo and sync submodules.
```bash
git clone https://github.com/apache/tvm.git
cd tvm
```

We use TVM version 0.10.0, checkout to the tag v0.10.0 and sync submodules.
```bash
git checkout v0.10.0
git submodule init
git submodule update
```

Execute the following commands to install the dependencies for TVM.
```bash
sudo apt-get update
sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
```

Create the build folder for TVM and copy the `config.cmake` file to the build folder.
```bash
mkdir build
cp cmake/config.cmake build
cd build
```

Configure and execute the build with the following commands (replace 4 with number of parallel compile jobs you need to run in your computer).
```bash
cmake ..
make runtime -j4
```

Now run the python installation to register TVM as a pip package.
```bash
cd ../python
python setup.py install --user
cd ..
```

Note that the `--user` flag is not necessary if you’re installing to a managed local environment, like `virtualenv`.

TVM is installed in the host now. You can check the instillation by executing the following command.
```bash
python -c "import tvm; print(tvm.__version__)"
```

---

### Clone and run the ACCELR TVM RISC-V Demo

Clone this repository to the shared directory between host and guest from the host side.
```bash
git clone https://github.com/accelr-net/tvm-riscv-demo.git
cd tvm-riscv-demo
```

Install the dependencies on host by executing the following commands.
```bash
pip install -r requirements.txt
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --index-url https://download.pytorch.org/whl/cpu
pip install onnx onnx2pytorch
sudo apt-get install python3-opencv
```

Execute the following command in host side to compile the models for host and the guest using TVM.
```bash
Explaination: python compile.py -a/-i/-k True -b/-x/-r True
Example: python compile.py -a True -b True
```

The command line arguments are explained below and can be executed accordingly.

```
  -i , --imagenet activate imagenet model, (True/False), Default: False
  -k , --kws activate kws model, (True/False), Default: False
  -a , --all_models activate all models, (True/False), Default: False
  -x , --x86_64 compile for x86_64 architecture, (True/False), Default: False
  -r , --riscv64 compile for riscv64 architecture, (True/False), Default: False
  -b , --both_archs compile for both architectures, (True/False), Default: False
```

Execute the following command in host side to run the host side half of the demo.
```bash
Explaination: python run.py -a/-i/-k/-m/-w True -s N
Example: python run.py -a True -s 5
```

The command line arguments are explained below and can be executed accordingly.
```
-a : All Models, (True/False), Default: False
-i : Imagenet TVM Model, (True/False), Default: False
-k : KWS TVM Model, (True/False), Default: False
-m : Imagenet Pytorch, Model (True/False), Default: False
-w : KWS Pytorch Model, (True/False), Default: False
-s : Number of test cases to be tested in Imagenet and KWS tests, (N: [-1,..n], N=-1 = All test cases available, N > -1: n test cases), Default: 10
```

Navigate to the repo which is in the shared directory in guest and install the dependencies on guest by executing the following commands.

```bash
pip install -r requirements.txt
sudo apt-get install python3-opencv

wget <PYTORCH_WHEEL_LINK>
wget <TORCHAUDIO_WHEEL_LINK>

sudo pip install torch-1.13.0a0+gitd1c1acd-cp38-cp38-linux_riscv64.whl
sudo pip install torchaudio-0.13.0+bc8640b-cp38-cp38-linux_riscv64.whl
```

Execute the same `python run.py -a True -s 5` like command in guest side to run the guest side half of the demo.

## Directory Stucture

```
.
└── tvm-riscv-demo
    ├── bin
    ├── compile.py
    ├── compiler
    │   └── compile.py
    ├── data
    │   ├── imagenet
    │   └── speechcommands
    ├── inference
    │   ├── dataloader
    │   │   ├── imagenet.py
    │   │   └── kws.py
    │   ├── models.py
    │   ├── session.py
    │   └── utils
    │       ├── evaluate.py
    │       └── helpers.py
    ├── LICENSE
    ├── models
    │   ├── lable.pickle
    │   ├── resnet18-kws-best-acc.pt
    │   ├── resnet18-v2-7.onnx
    │   └── synset.txt
    ├── README.md
    ├── requirements.txt
    └── run.py
```
