import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
import torchvision.models as models
model = models.alexnet().cuda()
inputs = torch.randn([2,3,224,224]).cuda()
times_profiler_cpu = []
times_profiler_gpu = []
def get_latency_from_string(s, sub_str= "CPU time total: "):
  index = s.find(sub_str)  # find the index of the substring
  if index != -1:  # check if substring is found
    print(s[-2])
    if s[-2] == "s" and s[-3:-1] != "ms":
      unit = "s"
      content = s[index + len(sub_str):-3]  # extract content following substring
    else:
       unit = s[-3:-1]
       content = s[index + len(sub_str):-3]  # extract content following substring
    #print(content)  # prints " jumps over the lazy dog"
    return content, unit
  else:
    print("Substring not found")
for i in range(10):
  with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)
  #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=1))
  time_gpu, unit_gpu = get_latency_from_string(prof.key_averages().table(sort_by="cpu_time_total", row_limit=1), "CUDA time total: ")
  #time_cpu, unit_cpu = get_latency_from_string(prof.key_averages().table(sort_by="cpu_time_total", row_limit=1))
  #times_profiler_cpu.append(float(time_cpu))
  times_profiler_gpu.append(float(time_gpu))
model = model.cpu()
inputs = inputs.cpu()
for i in range(10):
  with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)
  #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=1))
  time_cpu, unit_cpu = get_latency_from_string(prof.key_averages().table(sort_by="cpu_time_total", row_limit=1))
  times_profiler_cpu.append(float(time_cpu))
print("Mean time on cpu {} {}".format(np.mean(times_profiler_cpu),unit_cpu))
print("Standard deviation time on cpu {} {}".format(np.std(times_profiler_cpu), unit_cpu))
print("Mean time on gpu {} {}".format(np.mean(times_profiler_gpu),unit_gpu))
print("Standard deviation time on gpu {} {}".format(np.std(times_profiler_gpu), unit_gpu))

#Large Languase Model, such as llama2-7b.

