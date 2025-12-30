#!/bin/bash

# 可选：切换到 Python 脚本所在的目录
# cd /path/to/your/python/scripts

echo "开始执行 Python 脚本..."

# 顺序执行多个 Python 脚本
/home/lonqi/anaconda3/envs/CT2/bin/python simulation.py
/home/lonqi/anaconda3/envs/CT2/bin/python generate_data.py
/home/lonqi/anaconda3/envs/CT2/bin/python calibration.py

# 如果你想只有前一个成功（退出码为0）才执行下一个，使用 &&
# python3 script1.py && python3 script2.py && python3 script3.py

# 如果不管前一个是否成功都要继续执行，使用 ;
# python3 script1.py ; python3 script2.py ; python3 script3.py

echo "所有 Python 脚本执行完毕！"