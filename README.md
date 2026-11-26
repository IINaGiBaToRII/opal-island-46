# Build and test runs

This repository builds a command-line program (`crates/cli`, library `crates/engine`) for Linux and runs test jobs
on GitHub Actions runners. All rights reserved, no license is granted; see [NOTICE](NOTICE).

## Build
```
cargo build --release --locked -p runner        # binary: target/release/runner
```
The toolchain is pinned in `rust-toolchain.toml`. The **build** workflow (push to main, or manual) builds x86-64
(`-C target-cpu=x86-64-v3`) and aarch64 binaries, compares the result with `ci/canary.expected` (`ci/canary.sh`) and
publishes `runner-linux-x86_64` and `runner-linux-aarch64` to the `build-latest` release. `ci/EXPORT_ID` is the tree version.

## Test jobs (workflow match)
- The only input is `round`, an opaque 12-character id (0-9a-f); the run is named `r-<round>`.
- Everything going in and out is encrypted (`openssl enc -aes-256-cbc -pbkdf2 -iter 200000 -md sha256 -salt`) with the
  repository secret `ROUNDS_KEY`; encrypted files are named `<name>.<kfp>.enc`, where `kfp` is the first 12 hex digits of the key's sha256.
- Inputs are published in the `rounds-inputs` release: the manifest `m-<round>.<kfp>.enc` and input files, all `.enc`; the sha256 of each
  decrypted file is checked against the manifest. A mismatching key, a missing manifest or a different `ci/EXPORT_ID` fails the run.
- Outputs: artifact `shard-K` (`shard-K.<kfp>.enc`) and artifact `result` (`result.<kfp>.enc`).

Scripts: `ci/round.sh`, `ci/lib.sh`, `ci/manifest.py`, `ci/fetch_inputs.sh`, `ci/decrypt.sh`, `ci/shard.sh`, `ci/work.sh`,
`ci/merge.py`, `ci/canary.sh`, `ci/package.sh`.
