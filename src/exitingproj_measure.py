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
versions = [1,9]
#versions = [1,4,5,6,7]
#versions = [1,9,10,11,12]
version_names = ["baseline","GPU Final Design"]
#version_names = ["Baseline","LUT for 2 bits","Compact LUT for 2 bits","LUT for 1 bit","Compact LUT for 1 bit"]
#version_names = ["Baseline","Appending First and Last Rows","Stream Pipelining","Shared Mem for inboard","Shared Mem for LUT"]
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
    run_time_iter = []
    run_time_ws = []
    itera = 1
    for i in range(itera,itera+100002,20000):
        iteras.append(i)
        command = ['./gol', str(i), "inputs/1k.pbm", "outputs/1k.pbm"]
        print (command)
        stdout = subprocess.check_output(command).decode("utf-8")
        run_time_iter.append(float(stdout[18:]))
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
        run_time_ws.append(float(stdout[18:]))
        result.append(float(size*size)*it/float(stdout[18:])/1000000)
        size *= 2
    results.append(result)
#run cpu design, do make before run cpu design
gpusp_times = [[i/times[0][j.index(i)] for i in j] for j in times]
gpusp_results = [[i/results[0][j.index(i)] for i in j] for j in results]
cpuresult = [490.2057584,1487.297963,5605.008382,16408.15075,34279.14804]
cputime = [4.35896839,23270.81781,28687.14498,29939.29806,31301.2725,31383.52005]
cpusp_result=[1.0369512,1.439222116,1.437666761,1.442585131,1.441784147]
cpusp_time=[0.9480898761,1.446247093,1.46000002,1.509617581,1.531979318,1.578166499]
copybk = ["cp","life_opt.cu.bak","life_opt.cu"]
subprocess.run(copybk)
remove = ["rm","-f","life_opt.cu.bak"]
subprocess.run(remove)
print("gpu iter results ",times)
print("cpu iter results ",cputime)
print("cpu speedup iter results ",cpusp_time)
print(iteras)
print("gpu ws results ",results)
print("cpu ws results ",cpuresult)
print("cpu speedup ws results ",cpusp_result)
print(sizes)
print("gpu run_time_iter is:", run_time_iter)
print("gpu run_time_ws is:", run_time_ws)

for timedata in results[1:len(results)]:
    plt.plot(sizes, timedata, label= str(version_names[results.index(timedata)]))
plt.plot(sizes, cpuresult, label= "Existing Design")
plt.ylabel('Evaluated cells/sec in million')
plt.xlabel('World size')
plt.legend()
plt.show()
#plt.savefig("cps_ws.png")
for timedata in times[1:len(times)]:
    plt.plot(iteras, timedata, label= str(version_names[times.index(timedata)]))
plt.plot(iteras, cputime, label= "Existing Design")
plt.ylabel('Evaluated cells/sec in million')
plt.xlabel('Iteration')
plt.legend()
plt.show()
#plt.savefig("cps_iter.png")

for timedata in gpusp_results[1:len(gpusp_results)]:
    plt.plot(sizes, timedata, label= str(version_names[gpusp_results.index(timedata)]))
plt.plot(sizes, cpusp_result, label= "Existing Design")
plt.ylabel('Speed up')
plt.xlabel('World size')
plt.legend()
plt.show()
for timedata in gpusp_times[1:len(gpusp_times)]:
    plt.plot(iteras, timedata, label= str(version_names[gpusp_times.index(timedata)]))
plt.plot(iteras, cpusp_time, label= "CPU Design")
plt.ylabel('Speed up')
plt.xlabel('Iteration')
plt.legend()
plt.show()
