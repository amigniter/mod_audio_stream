# mod_audio_stream (internal fork)

Internal fork of [`amigniter/mod_audio_stream`](https://github.com/amigniter/mod_audio_stream) — a FreeSWITCH module that streams call audio to a WebSocket server and receives responses (audio or JSON) back.

This repository exists so we can track the exact version running in production, capture any local build/config changes in version control, and have a clean place to apply patches or migrate off the upstream module if needed.

---

## Current production baseline

| Item | Value |
|---|---|
| Upstream repo | https://github.com/amigniter/mod_audio_stream |
| Upstream tag | `v1.0.3` |
| Upstream commit | `b13ccbb` ("forward stream imprv (stream buffer size)") |
| Local code changes | None (only `chmod +x` on `build-mod-audio-stream.sh`) |
| WebSocket library used by v1.0.3 | `libwsc` (closed-source, shipped by upstream) |
| Production server path | `/usr/src/mod_audio_stream` on `freeswitch-prod` |

---

## ⚠️ Critical: 10 concurrent streams limit in v1.0.3

**v1.0.3 has a hard cap of 10 concurrent streaming channels.** This is **not** a bug, a config issue, or something we can patch from this repo. It is a licensing restriction enforced by the upstream maintainer.

From the upstream README:

> This release is a commercial product that is available for free use, including commercial use, with a limitation of 10 concurrent streaming channels. For users requiring more than 10 channels, or access to the source code, please contact us for further information and licensing options.

### Why we can't just patch it

Starting in v1.0.3, upstream replaced the previous WebSocket library (`ixwebsocket`) with their own in-house client called **`libwsc`**. `libwsc` is **closed source** — it is not in the GitHub repo and we do not have access to it. The 10-channel cap is enforced inside `libwsc` (or in how `mod_audio_stream` calls into it). Since the open-source portion of this repo only contains the FreeSWITCH glue code and links against a prebuilt `libwsc` binary, there is nothing in this fork we can modify to lift the cap.

### What we can do about it

There are four realistic paths forward. We have not chosen one yet — see the TODO list at the bottom.

1. **Buy a commercial license from upstream (amigniter).** Cleanest path. Get either a higher cap or full `libwsc` source. Contact via the upstream GitHub repo. Worth getting a quote even if we don't proceed.
2. **Downgrade to v1.0.2 or earlier.** Pre-v1.0.3 used `ixwebsocket` and is fully open source with no channel cap. We lose the v1.0.3 improvements (stream buffer handling, libwsc latency optimizations) but gain unlimited concurrency and full source ownership.
3. **Migrate to [`mod_audio_fork`](https://github.com/drachtio/drachtio-freeswitch-modules)** by Dave Horton. This is the original module that inspired `mod_audio_stream`. Fully open source, no cap, widely deployed in production. Migration is real work but bounded — the WebSocket protocol contract is similar.
4. **Fork v1.0.2 ourselves and maintain it long-term.** Take the last fully open version and own it. No upstream upgrades after that point, but full control.

---

## TODO — verification work before we commit to a path

These items need to be done before we make a licensing/architecture decision. Assignee TBD.

- [ ] **Confirm the version actually loaded in FreeSWITCH matches `/usr/src/mod_audio_stream`.** Run on `freeswitch-prod`:
  ```bash
  fs_cli -x "module_exists mod_audio_stream"
  fs_cli -x "show modules" | grep audio_stream
  ls -la /usr/local/freeswitch/mod/mod_audio_stream.so
  ```
  Compare the `.so` file's mtime against the build date in `/usr/src/mod_audio_stream/build/`.

- [ ] **Measure current peak concurrent streams in production.** We need to know whether we are already hitting the cap or not. During peak hours run:
  ```bash
  fs_cli -x "show channels" | grep -c .
  # And inspect mod_audio_stream-specific session counts via logs or:
  fs_cli -x "show channels" | grep -i stream
  ```
  Log this over a 24–48 hour window. If we are routinely at or near 10, we have an active production problem, not a future one.

- [ ] **Find out what the cap actually does when hit.** Does the 11th call fail to start streaming? Drop silently? Hang up? Return an error to the dialplan? This determines our blast radius. Test in staging by forcing 11+ concurrent streams against a dummy WebSocket sink.

- [ ] **Get a commercial license quote from amigniter.** Email/issue on the upstream repo. We need a number to compare against the engineering cost of options 2–4.

- [ ] **Audit v1.0.2 source** to confirm it is cap-free and uses `ixwebsocket`. Check the `v1.0.2` tag in the upstream repo and verify there is no equivalent restriction.

- [ ] **Scope a `mod_audio_fork` migration.** How different is the WebSocket message format? What channel variables would need to change in our dialplan? Estimate the work in days.

- [ ] **Delete the leftover `/usr/src/libwebsockets` directory on `freeswitch-prod`** once we confirm nothing on the box still links against it. v1.0.3 does not use libwebsockets at all (it uses libwsc, and earlier versions used ixwebsocket). That directory is dead weight from an older build attempt and is misleading anyone who reads the box.

---

## Build dependencies

Per upstream v1.0.3:

```bash
sudo apt-get install -y \
    libfreeswitch-dev \
    libssl-dev \
    zlib1g-dev \
    libevent-dev \
    libspeexdsp-dev \
    git \
    cmake \
    build-essential
```

Note: **libevent**, not libwebsockets. `libwsc` is libevent-based and is fetched/linked as part of the upstream build process — we do not install or maintain it separately.

## Build steps

```bash
git clone <this-repo-url> mod_audio_stream
cd mod_audio_stream
chmod +x build-mod-audio-stream.sh
sudo ./build-mod-audio-stream.sh
```

The build script runs `cmake` in a `build/` subdirectory, compiles, and installs `mod_audio_stream.so` into the FreeSWITCH module path (typically `/usr/local/freeswitch/mod/`).

After install, load it in FreeSWITCH:

```bash
fs_cli -x "load mod_audio_stream"
```

To load it on every FreeSWITCH start, add to `/usr/local/freeswitch/conf/autoload_configs/modules.conf.xml`:

```xml
<load module="mod_audio_stream"/>
```

---

## Usage (quick reference)

Start streaming a channel's audio to a WebSocket server from the dialplan or via `fs_cli`:

```
uuid_audio_stream <uuid> start ws://your-ws-server:port/path mixed 8k metadata-string
```

Stop streaming:

```
uuid_audio_stream <uuid> stop
```

Channel variables for tuning (TLS, deflate, heartbeat, suppress logging, buffer size) are documented in the upstream README: https://github.com/amigniter/mod_audio_stream

---

## Repository hygiene notes

- This repo contains **only** the `mod_audio_stream` source. We do **not** vendor `libwebsockets`, `libwsc`, or `libevent` — they are external dependencies installed at build time on the target machine.
- Build artifacts (`build/`, `*.o`, `*.so`, `*.la`, `.libs/`, `CMakeCache.txt`, `CMakeFiles/`) must not be committed. See `.gitignore`.
- If we ever apply local patches, each one should be a separate commit with a clear message explaining **why**, and ideally referencing a ticket or upstream issue.
- To pull updates from upstream later:
  ```bash
  git remote add upstream https://github.com/amigniter/mod_audio_stream.git
  git fetch upstream
  git log HEAD..upstream/main --oneline   # see what's new
  ```

---

## History of this fork

- **<DATE>** — Repo created from `/usr/src/mod_audio_stream` on `freeswitch-prod` at upstream tag `v1.0.3` (commit `b13ccbb`). No local code changes carried over. Created to (a) version-control the production baseline, and (b) give us a place to land a fix or migration for the 10-concurrent-streams cap.
