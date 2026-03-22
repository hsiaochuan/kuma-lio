a_fname = '/tmp/result_2026-03-22-22-35-04.txt'
b_fname = '/tmp/result_2026-03-22-23-07-34.txt'

af = open(a_fname, 'r')
bf = open(b_fname, 'r')
a = dict()
b = dict()
for line_num, line in enumerate(af, 1):
    if ',' not in line:
        continue
    line = line.strip()
    parts = line.split(',', 1)
    name = parts[0].strip()
    num_str = parts[1].strip()
    num = float(num_str)
    a[name] = num
for line_num, line in enumerate(bf, 1):
    if ',' not in line:
        continue
    line = line.strip()
    parts = line.split(',', 1)
    name = parts[0].strip()
    num_str = parts[1].strip()
    num = float(num_str)
    b[name] = num

result_f = open('/tmp/compare.txt', 'w')
for key, value in a.items():
    a_v = value
    b_v = b[key]
    ratio = (b_v - a_v) / a_v * 100
    result_f.write('{},\t{:.3f}%\n'.format(key, ratio))
