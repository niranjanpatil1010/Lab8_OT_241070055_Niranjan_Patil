"""Drive xv6-riscv under QEMU: boot, run shell commands, capture output.
Usage: python3 xv6run.py <xv6_dir> <cmd> [<cmd>...]   (cmds run sequentially in one boot)
Each command's output is printed between markers. Exit code 0 if boot reached the shell.
"""
import subprocess, sys, os, time, select, shutil, tempfile

def run(xv6_dir, cmds, cpus=1, timeout=900, quiet=False):
    # boot from a private copy of the disk image so concurrent runs never share state
    tmpimg = tempfile.NamedTemporaryFile(suffix=".img", delete=False).name
    shutil.copyfile(os.path.join(xv6_dir, "fs.img"), tmpimg)
    qemu = ["qemu-system-riscv64", "-machine", "virt", "-bios", "none",
            "-kernel", os.path.join(xv6_dir, "kernel/kernel"), "-m", "128M",
            "-smp", str(cpus), "-nographic",
            "-global", "virtio-mmio.force-legacy=false",
            "-drive", f"file={tmpimg},if=none,format=raw,id=x0",
            "-device", "virtio-blk-device,drive=x0,bus=virtio-mmio-bus.0"]
    p = subprocess.Popen(qemu, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, bufsize=0)
    buf = b""
    outs = []
    t0 = time.time()

    def read_until_prompt():
        nonlocal buf
        while True:
            if time.time() - t0 > timeout:
                raise TimeoutError("xv6 timeout; tail=" + buf[-300:].decode(errors="replace"))
            r, _, _ = select.select([p.stdout], [], [], 1.0)
            if r:
                d = os.read(p.stdout.fileno(), 65536)
                if not d: raise EOFError("qemu exited; tail=" + buf[-300:].decode(errors="replace"))
                buf += d
                if buf.endswith(b"$ "):
                    s = buf.decode(errors="replace"); buf = b""
                    return s
    try:
        boot = read_until_prompt()
        for c in cmds:
            p.stdin.write((c + "\n").encode()); p.stdin.flush()
            o = read_until_prompt()
            outs.append(o)
    finally:
        p.kill(); p.wait(); os.unlink(tmpimg)
    return boot, outs

if __name__ == "__main__":
    d = sys.argv[1]; cmds = sys.argv[2:]
    boot, outs = run(d, cmds)
    print(boot)
    for c, o in zip(cmds, outs):
        print(f"##### {c}\n{o}")
