#!/bin/bash
cd "$(dirname "$0")"
python3 run.py 10 100 > logs/bench.out 2>&1
for m in rr lottery; do
 python3 -c "
import xv6run,time
t=time.time(); b,o=xv6run.run('build/$m',['usertests'],timeout=3000)
open('logs/usertests_patched_$m.txt','w').write(o[0]+'\nwall_s=%.1f\n'%(time.time()-t))" > logs/ut_$m.out 2>&1 &
done; wait; echo ALLDONE > logs/done.flag
