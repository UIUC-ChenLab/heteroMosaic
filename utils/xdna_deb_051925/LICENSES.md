# License and source information

This directory contains unmodified AMD XRT and XDNA Debian packages for
Ubuntu 24.04 on `amd64`. It is an unofficial mirror and is not endorsed by AMD.

## XRT packages

The `xrt-base`, `xrt-base-dev`, and `xrt-npu` packages contain components
licensed under Apache-2.0, GPL-2.0, MIT, and other compatible third-party
licenses. The authoritative XRT license and attribution notices are included
as [`LICENSE-XRT`](LICENSE-XRT) and [`NOTICE-XRT`](NOTICE-XRT), and are also
embedded in `xrt-base` at:

- `/opt/xilinx/xrt/license/LICENSE`
- `/opt/xilinx/xrt/share/doc/NOTICE`

Upstream XRT source commit:
<https://github.com/Xilinx/XRT/tree/ce38ec8e3150d76b80862e43a2e695022ebed787>

## XDNA plugin and firmware

The `xrt_plugin-amdxdna` package contains the XRT SHIM, DKMS driver source,
firmware, and validation binaries. Open-source components retain their
per-file SPDX licenses. AMD binary components are governed by
[`LICENSE.amdnpu`](LICENSE.amdnpu), which permits redistribution only in
unmodified binary form and requires reproducing its copyright and license
notice.

Upstream XDNA source commit:
<https://github.com/amd/xdna-driver/tree/0e6d303b2cc2b3fe1cf10aba0acbf57a422588fb>

No package in this directory has been modified.
