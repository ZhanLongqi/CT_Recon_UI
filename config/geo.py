from tigre import geometry
import numpy as np
geo = geometry()
geo.mode = 'cone'  # 设定为锥束模式
geo.DSD = 1105.12 * 0.2  # 源到探测器距离
geo.DSO = geo.DSD / 2  # 源到物体中心距离
geo.nDetector = np.array([1,384],dtype=np.int32)  # 探测器像素数
geo.nVoxel = np.array([1,384,384])
geo.dVoxel = np.array([0.1,0.1,0.1],dtype=np.float32)
geo.dDetector = np.array([0.2, 0.2],dtype=np.float32)  # 单个探测器像素大小
geo.sDetector = geo.nDetector * geo.dDetector
geo.angles = np.linspace(0, 2 * np.pi, 600) 
geo.offDetector = np.array([0,-8.9*0.2],dtype=np.float32)
geo.offOrigin = np.array([-geo.nVoxel[0]/2*geo.dVoxel[0],-geo.nVoxel[1]/2*geo.dVoxel[1],-geo.nVoxel[2]/2*geo.dVoxel[2]],np.float32)#z,y,x cm
geo.rotDetector = np.array([0,0,0])