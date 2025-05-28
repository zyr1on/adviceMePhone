import torch
print(torch.__version__)
print("CUDA aktif mi? ➤", torch.cuda.is_available())
print("Kullanılan cihaz ➤", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
