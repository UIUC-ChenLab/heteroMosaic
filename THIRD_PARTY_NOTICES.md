# Third-Party Notices

heteroMosaic is distributed under the Apache License 2.0. The third-party
components listed below remain subject to their respective licenses. This file
is informational; the complete license texts referenced below control.

## BS::thread_pool

- Version: 5.0.0
- Upstream: https://github.com/bshoshany/thread-pool/tree/v5.0.0
- Vendored file: `include/third_party/BSpool/BS_thread_pool.hpp`
- License: MIT
- Copyright: Copyright (c) 2024 Barak Shoshany
- License text: `include/third_party/BSpool/LICENSE.txt`
- Local changes: whitespace-only formatting; no semantic changes from the
  v5.0.0 upstream header.

The upstream author requests a link to the project in source code and
documentation. For published research, the author also requests citation of:

Barak Shoshany, "A C++17 Thread Pool for High-Performance Scientific
Computing", SoftwareX 26 (2024) 101687,
https://doi.org/10.1016/j.softx.2024.101687.

## JSON for Modern C++ (nlohmann/json)

- Version: 3.11.3
- Upstream: https://github.com/nlohmann/json/tree/v3.11.3
- Vendored file: `include/third_party/nlohmann/json.hpp`
- Primary license: MIT
- Copyright: Copyright (c) 2013-2023 Niels Lohmann
- Primary license text: `include/third_party/nlohmann/LICENSE.MIT`
- Local changes: none; the vendored header is byte-for-byte identical to the
  v3.11.3 single-header release.

As documented by upstream, the single-header library also contains:

- A UTF-8 decoder by Bjoern Hoehrmann, copyright 2008-2009, under MIT.
- A modified Grisu2 implementation by Florian Loitsch, copyright 2009, under
  MIT.
- Hedley code by Evan Nemerson under CC0-1.0.
- Portions of Google Abseil under Apache-2.0.

The corresponding license texts are retained in
`include/third_party/nlohmann/LICENSES/`.
