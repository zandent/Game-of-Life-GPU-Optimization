#!/usr/bin/python3
import subprocess
import shlex
import sys
import matplotlib.pyplot as plt
import re
times = []
copy = ["cp","life_opt.cu","life_opt.cu.bak"]
subprocess.run(copy)
for ver in range(1,9):
    with open("life_opt.cu","w") as writer:
        with open("life_opt.cu.bak",'r') as reader:
            rdline = reader.readlines()
        #print (rdline)
        for line in rdline:
            version = re.search(r'\#define GPU\_IMPL\_VERSION',line)
            if version:
                print (version.group())
                writer.write("#define GPU_IMPL_VERSION "+str(ver)+'\n')
            else:
                writer.write(line) 
    subprocess.run(["make"])
    time = []
    iteras = []
    itera = int(sys.argv[1])
    for i in range(itera,itera+20000,1000):
        iteras.append(i)
        command = ['./gol', str(i), sys.argv[2], sys.argv[3]]
        print (command)
        stdout = subprocess.check_output(command).decode("utf-8")
        time.append(float(stdout[18:]))
    times.append(time)
remove = ["rm","-f","life_opt.cu.bak"]
subprocess.run(remove)
print(times)
print(iteras)
for timedata in times:
    plt.plot(iteras, timedata, label="Version " + str(times.index(timedata)+1))
plt.ylabel('Time')
plt.xlabel('Iteration')
plt.legend()
plt.show()
