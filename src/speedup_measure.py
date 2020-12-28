#!/usr/bin/python3
import subprocess
import shlex
import sys
import matplotlib.pyplot as plt
import re
times = []
results = []
copy = ["cp","life_opt.cu","life_opt.cu.bak"]
subprocess.run(copy)
#NOTE: versions list must be start with 1
#versions = [1,2,3,5,8,9]
#version_names = ["Baseline","Bit Representation","LUT for 4 bits","Compact LUT for 2 bits","Pinned Memory","Appending First and Last Rows"]
#version_names = ["Baseline","LUT for 2 bits","Compact LUT for 2 bits","LUT for 1 bit","Compact LUT for 1 bit"]
version_names = ["Baseline","Appending First and Last Rows","Stream Pipelining","Shared Mem for inboard","Shared Mem for LUT"]
#versions = [1,4,5,6,7]
versions = [1,9,10,11,12]
for ver in versions:
    with open("life_opt.cu","w") as writer:
        with open("life_opt.cu.bak",'r') as reader:
            rdline = reader.readlines()
        #print (rdline)
        for line in rdline:
            version = re.search(r'\#define GPU\_IMPL\_VERSION',line)
            if version:
                #print (version.group())
                writer.write("#define GPU_IMPL_VERSION "+str(ver)+'\n')
            else:
                writer.write(line) 
    subprocess.run(["make"])
    time = []
    iteras = []
    itera = 1
    for i in range(itera,itera+100002,20000):
        iteras.append(i)
        command = ['./gol', str(i), "inputs/1k.pbm", "outputs/1k.pbm"]
        print (command)
        stdout = subprocess.check_output(command).decode("utf-8")
        time.append(float(1048576)*i/float(stdout[18:])/1000000)
    times.append(time)
    result = []
    sizes = []
    size = 128
    it = 10000
    for i in range(1,6):
        sizes.append(size*size)
        init = ['./initboard',str(size),str(size),'inputs/'+str(size)+'.pbm']
        print (init)
        subprocess.run(init)
        command = ['./gol', str(it), 'inputs/'+str(size)+'.pbm','outputs/'+str(size)+'.pbm']
        print (command)
        stdout = subprocess.check_output(command).decode("utf-8")
        result.append(float(size*size)*it/float(stdout[18:])/1000000)
        size *= 2
    results.append(result)
copybk = ["cp","life_opt.cu.bak","life_opt.cu"]
subprocess.run(copybk)
remove = ["rm","-f","life_opt.cu.bak"]
subprocess.run(remove)
times = [[i/times[0][j.index(i)] for i in j] for j in times]
results = [[i/results[0][j.index(i)] for i in j] for j in results]
print(times)
print(iteras)
print(results)
print(sizes)
for timedata in results[1:len(results)]:
    plt.plot(sizes, timedata, label= str(version_names[results.index(timedata)]))
plt.ylabel('Speed up')
plt.xlabel('World size')
plt.legend()
plt.show()
for timedata in times[1:len(times)]:
    plt.plot(iteras, timedata, label= str(version_names[times.index(timedata)]))
plt.ylabel('Speed up')
plt.xlabel('Iteration')
plt.legend()
plt.show()
