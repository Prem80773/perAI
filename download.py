import cv2

sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel("EDSR_x4.pb")  # You need to download .pb once from OpenCV GitHub
sr.setModel("edsr", 4)

image = cv2.imread("path.jpg")
upscaled = sr.upsample(image)
cv2.imwrite("upscaled.jpg", upscaled)
