

def fix_cpu_freq(freq, n_cpu=8):
    dir = "/sys/devices/system/cpu/"
    f_names = ["scaling_max_freq", "scaling_min_freq"]
    for i in range(n_cpu):
        path = "{}cpu{}/cpufreq/".format(dir, i)
        for f_name in f_names:
            with open(path+f_name, "wt") as f:
                f.write(str(freq))
        with open(path+"scaling_cur_freq") as f:
            print("{} cur_freq: {}".format(i, f.read()), end="")


if __name__ == "__main__":
    fix_cpu_freq(3000000)