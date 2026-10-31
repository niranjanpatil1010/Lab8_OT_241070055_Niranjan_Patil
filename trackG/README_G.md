# Track G — xv6 lottery scheduler (how to reproduce)
Base: https://github.com/mit-pdos/xv6-riscv @ commit e93dabeb98431235c6e1a101cb96c73c923fdbc4. Tools: qemu-system-riscv64 8.2, riscv64-linux-gnu-gcc 13.
```
git clone https://github.com/mit-pdos/xv6-riscv && cd xv6-riscv && git checkout e93dabe
git apply /path/to/trackG/xv6_patch.diff
# baseline (round-robin, with the same accounting code) and lottery kernels:
make clean; make TOOLPREFIX=riscv64-linux-gnu- CPUS=1 kernel/kernel fs.img
mkdir -p ../optimize-anything/trackG/build/rr/kernel && cp kernel/kernel ../optimize-anything/trackG/build/rr/kernel/ && cp fs.img ../optimize-anything/trackG/build/rr/
make clean; make TOOLPREFIX=riscv64-linux-gnu- SCHED=lottery CPUS=1 kernel/kernel fs.img
mkdir -p ../optimize-anything/trackG/build/lottery/kernel && cp kernel/kernel ../optimize-anything/trackG/build/lottery/kernel/ && cp fs.img ../optimize-anything/trackG/build/lottery/
# interactive:  make TOOLPREFIX=riscv64-linux-gnu- SCHED=lottery CPUS=1 qemu    then  $ schedbench prop 100 1   /  $ usertests
cd ../optimize-anything && python3 trackG/run.py 10 100      # 10 runs x 100 ticks -> results/trackG_{raw.txt,results.json}
```
`xv6run.py` boots QEMU from a private copy of fs.img and drives the shell over stdin/stdout.
