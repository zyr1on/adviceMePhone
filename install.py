import os
import platform
import subprocess


system_platform = platform.system()
requirements_file = 'requirements.txt'

if system_platform == 'Windows':
    pip_command = ['pip.exe', 'install', '-r', requirements_file]
elif system_platform == 'Linux' or system_platform == 'Darwin':  # MacOS 'Darwin' olarak tanımlanır
    pip_command = ['pip3', 'install', '-r', requirements_file]
else:
    raise EnvironmentError(f"Desteklenmeyen işletim sistemi: {system_platform}")

try:
    subprocess.check_call(pip_command)
    print("\nBağımlılıklar başarıyla yüklendi.\n")
except subprocess.CalledProcessError as e:
    print(f"Bağımlılıkları yüklerken bir hata oluştu: {e}")


if not os.path.exists("model.pt"):
    import gdown
    url = 'https://drive.google.com/uc?export=download&id=1vtK37taMctc1mMG84Mx1OHN0DItJY3hW'
    output= 'model.pt'
    print("Model dosyası yükleniyor.")
    gdown.download(url, output, quiet=False, proxy=False)
    print("Model dosyası yüklendi.")
    print("python3 app.py ile programı çalıştırabilirsiniz.")
else:
    print("Model dosyası zaten mevcut\n 'python3 app.py' ile programı çalıştırabilirsiniz.")
