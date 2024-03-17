import numpy as np

def hog_özellikleri_benzerlikleri(hog1, hog2):
    öklid = np.linalg.norm(hog1 - hog2)
    
    skor = 1 / (1 + öklid)
    
    return skor

hog1 = np.loadtxt(r"C:\Users\cypoi\Masaüstü\VisionaryTracker\hog\single_track\hog_features\frame_9.txt")
hog2 = np.loadtxt(r"C:\Users\cypoi\Masaüstü\VisionaryTracker\hog\single_track\hog_features\frame_10.txt")

skor = hog_özellikleri_benzerlikleri(hog1, hog2)
print("Benzerlik Skoru:", skor)

