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

<style>
  .banner {
    background-color: #f0f0f0;
    padding: 20px;
    border: 1px solid #ccc;
    border-radius: 5px;
    text-align: Left;
  }

  .banner h4 {
    color: #333;
  }

  .banner p {
    font-style: italic;
  }
</style>


## Introduction
This repository contains the demo application we have created to run and benchmark Image Classification and KWS (Key Word Spotting) models on a [QEMU](https://www.qemu.org/) emulated [RV64IMFD](https://www3.diism.unisi.it/~giorgi/didattica/tools1/RISCV_ISA_TABLE-v11.pdf) Ubuntu Linux system through [Apache TVM](https://tvm.apache.org/).

## Getting Started
Fist follow the instructions [here](./docs/ubuntu_qemu.md)  prepare a QEMU emulated RISC-V environment on an x86_64 system. *For the rest of the guide, x86_64 system will be called the host, and RISC-V system will be called the guest.*

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

<div class="banner">
  <h4>Note:</h4>
  <p>
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
  </p>
</div>

#### a