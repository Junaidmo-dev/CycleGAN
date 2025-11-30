import sys, os, traceback
# add img2img_turbo path
# script is in backend/, img2img_turbo is in backend/img2img_turbo
base = os.path.join(os.path.dirname(__file__), 'img2img_turbo')
sys.path.append(base)
print('sys.path added', base)
try:
    from src.pix2pix_turbo import Pix2Pix_Turbo
    print('Import succeeded')
except Exception as e:
    print('Import failed')
    traceback.print_exc()
