# Running Ubuntu on a QEMU emulated RISC-V system

First create a workspace folder with a name similar to riscv-ubuntu and do the cloning of all the rest of repos to within that workspace folder.

Run `sudo apt update` & `sudo apt upgrade` before going through the following steps!

---

### 1. Clone and build RISC-V GNU Toolchain
```bash
git clone https://github.com/riscv-collab/riscv-gnu-toolchain --recursive
cd riscv-gnu-toolchain
git checkout <latest release tag>
./configure --prefix=/opt/riscv
make linux -j $(nproc)
```
__Note:__ You need to create `/opt/riscv` directory first which is the installation directory for riscv gcc.
since it's a root directory either you need to run sudo make linux ... or provide the user permission for the directory by executing `sudo chown -R <user>:<user> /opt/riscv`.

---

### 2. Install required dependencies for the process
```bash
sudo apt install ninja-build libglib2.0-dev qemu-system-misc qemu-utils bison flex libssl-dev
```

---

### 3. Clone and build QEMU
```bash
git clone https://github.com/qemu/qemu
cd qemu
git checkout <latest release tag>
./configure --target-list=riscv64-softmmu --enable-slirp
make -j $(nproc)
sudo make install
```

---

### 4. Clone and build U-Boot
```bash
git clone https://source.denx.de/u-boot/u-boot.git
cd u-boot
git checkout v2023.07.02
export CROSS_COMPILE=riscv64-unknown-linux-gnu-
export PATH=/opt/riscv/bin:$PATH
make qemu-riscv64_smode_defconfig
make -j$(nproc)
```

---

### 5. Clone and build OpenSBI
```bash
git clone https://github.com/riscv-software-src/opensbi.git
cd opensbi
git checkout v1.2
```

Execute the following command only if you are continuing the process in a different terminal session from the session you started the process.

```bash
export CROSS_COMPILE=riscv64-unknown-linux-gnu-
export PATH=/opt/riscv/bin:$PATH
```

Now execute the following commands assuming that u-boot repo is in parallel with opensbi repo.

```bash
make PLATFORM=generic FW_PAYLOAD_PATH=../u-boot/u-boot.bin -j$(nproc)
```

---

### 6. Download and prepare Ubuntu image
```bash
# downloading the disk image
wget https://cdimage.ubuntu.com/releases/20.04.6/release/ubuntu-20.04.5-preinstalled-server-riscv64+unmatched.img.xz
# unpacking/decompressing the disk image
xz -dk ubuntu-20.04.5-preinstalled-server-riscv64+unmatched.img.xz
# renaming the disk image
mv ubuntu-20.04.5-preinstalled-server-riscv64+unmatched.img ubuntu.img
# adding 8GB to the disk
qemu-img resize -f raw ubuntu.img +8G
```

---

### 7. Run QEMU with RISC-V and Ubuntu
```bash
qemu-system-riscv64 -machine virt -nographic \
    -m 4.0G -smp 2 \
    -bios ./opensbi/build/platform/generic/firmware/fw_payload.elf \
    -device virtio-net-device,netdev=eth0 \
    -netdev user,id=eth0,hostfwd=tcp::5555-:22 \
    -drive file=ubuntu.img,format=raw,if=virtio
```

---

### 8. Mount host folder in guest VM (extra step)
Add the following option to run qemu command.
```bash
-virtfs local,path=<host local path to shared folder>,mount_tag=<tag name>,security_model=none
```
After the guest VM is up, mount the folder using the following command.
```bash
sudo mkdir /mnt/<tag name>
sudo mount -t 9p -o trans=virtio,version=9p2000.L <tag name> /mnt/<folder name>
```

---
