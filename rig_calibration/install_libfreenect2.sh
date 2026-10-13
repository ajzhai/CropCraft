#!/usr/bin/env bash
# Build and install libfreenect2 (Kinect v2 driver) from source.
# Ubuntu has no libfreenect2 package -- only libfreenect (Kinect v1) -- so this
# has to be built. Run this yourself in a real terminal (not via an automated
# tool) since it needs sudo for apt-get, udev rules, and reload.
set -euo pipefail

echo "== installing build dependencies =="
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config git \
    libusb-1.0-0-dev libturbojpeg0-dev libglfw3-dev libjpeg-dev

# Optional: GPU-accelerated depth decoding via OpenCL. Not required -- the
# CPU pipeline works without it, just slower per frame. Uncomment if you
# want it and know your GPU's ICD package name.
# sudo apt-get install -y ocl-icd-opencl-dev opencl-headers intel-opencl-icd

echo "== cloning + building libfreenect2 =="
INSTALL_PREFIX="$HOME/freenect2"
SRC_DIR="$HOME/libfreenect2"

if [ ! -d "$SRC_DIR" ]; then
    git clone https://github.com/OpenKinect/libfreenect2.git "$SRC_DIR"
fi
cd "$SRC_DIR"
mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX"
make -j"$(nproc)"
make install

echo "== installing udev rules (so non-root access works) =="
sudo cp "$SRC_DIR/platform/linux/udev/90-kinect2.rules" /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger

cat <<EOF

Build complete: $INSTALL_PREFIX

Next: physically unplug and replug both Kinect v2 units (udev rules only
apply to newly-enumerated devices), then test each one individually:

  $INSTALL_PREFIX/bin/Protonect --help          # lists connected device serials
  $INSTALL_PREFIX/bin/Protonect cpu <serial-A>   # should open a live RGB+depth window
  $INSTALL_PREFIX/bin/Protonect cpu <serial-B>

Confirm BOTH show live, correctly-oriented RGB and depth streams before
starting any real plant capture -- this is the actual readiness gate, the
USB-level detection I already checked is necessary but not sufficient.

For scripted (non-interactive) capture from Python, add the bindings:
  pip install --user git+https://github.com/r9y9/pylibfreenect2.git
(you'll need to point it at $INSTALL_PREFIX via LIBFREENECT2_INSTALL_PREFIX
as described in that repo's README).

Before running BOTH Kinects at once, also raise the usbfs DMA memory limit
(default 16MB is only enough for one device) -- see the "usbfs DMA memory
limit" section in dataset/README.md.
EOF

EOF
